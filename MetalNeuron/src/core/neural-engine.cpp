#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>

#include "neural-engine.h"
#include "multi-layer-kernels.h"
#include "dropout-layer.h"
#include "mnist-dataset.h"
#include "training-manager.h"
#include "math-lib.h"
#include "layer-factory.h"


#ifdef DEBUG_GRADIENT_CHECKS
#include "debug/gradient-checker.h"
#endif


const char* outputFileName = "multilayer_nn_training.m";
int globalTimestep = 0;


NeuralEngine::NeuralEngine(MTL::Device* pDevice, const ModelConfig& config, DataManager* pDataManager)
: _pDevice(pDevice->retain()),
areBuffersBuilt(false),
currentlyComputing(false),
_pDataManager(pDataManager),
_pInputLayer(nullptr),
input_dim(0),
output_dim(0),
epochs(0),
filename(config.filename)
{
    batch_size = config.training.batch_size;
    epochs = config.training.epochs;
    
    input_dim = _pDataManager->inputDim();
    output_dim = _pDataManager->outputDim();
    
    _pLogger = new Logger(outputFileName, config.dataset.type == "function");
    
    _pKeyboardController = new KeyboardController();
    
    _pKeyboardController->setForwardCallback([this, config]() {
        _pLogger->clear();
        TrainingManager::instance().setTraining(false);
        
        computeForwardBatches(_pDataManager->getCurrentDataset()->numSamples(), ceil((float)_pDataManager->getCurrentDataset()->numSamples() / config.training.batch_size), [this]() {
            std::cout << "✅ Forward pass complete!" << std::endl;
        });
    });
    
    _pKeyboardController->setLearnCallback([this, config]() {
        _pLogger->clear();
        TrainingManager::instance().setTraining(true);
        
        auto currentEpoch = std::make_shared<int>(0);
        
        // Define epoch callback shared_ptr for recursion
        auto epochCallback = std::make_shared<std::function<void()>>(); //FIXME this is too flowery and complicated for me
        
        *epochCallback = [this, currentEpoch, epochCallback, config]() {
            if (*currentEpoch >= epochs) {
                std::cout << "✅ Training complete!" << std::endl;
                return;
            }
            
            std::cout << "🔄 Starting epoch: " << (*currentEpoch + 1) << " / " << epochs << std::endl;
            
            // Run batches for the current epoch, and then call next epoch on completion
            computeBackwardBatches(_pDataManager->getCurrentDataset()->numSamples(), ceil((float)_pDataManager->getCurrentDataset()->numSamples() / config.training.batch_size), [this, currentEpoch, epochCallback]() {
                (*currentEpoch)++;
                (*epochCallback)();
            });
        };
        
        // Start the epoch processing
        (*epochCallback)();
    });
    
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _pKeyboardController->setSaveCallback([this]() {
        saveModel(filename + ".bin");
    });
    
    _pKeyboardController->setLoadCallback([this]() {
        loadModel(filename + ".bin");
    });
    
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    
    buildComputePipeline();
    
    createDynamicLayers(config);
}

NeuralEngine::~NeuralEngine() {
    for (auto layer : dynamicLayers_)
        delete layer;
    
    delete _pKeyboardController;
    delete _pLogger;
    
    if (_pCommandQueue) _pCommandQueue->release();
    if (_pDevice) _pDevice->release();
}


void NeuralEngine::createDynamicLayers(const ModelConfig& config) {
    // Clear existing layers
    for (auto layer : dynamicLayers_) {
        delete layer;
    }
    dynamicLayers_.clear();
    
    int first_layer_time_steps = config.first_layer_time_steps > 0 ? config.first_layer_time_steps : 1;
    
    input_dim = _pDataManager->inputDim();
    output_dim = _pDataManager->outputDim();
    
    _pInputLayer = new InputLayer(input_dim, first_layer_time_steps);
    
    _pDataManager->initialize([this, config]() {
        _pLogger->clear();
        buildBuffers();
        connectDynamicLayers(config);
    });
}

void NeuralEngine::connectDynamicLayers(const ModelConfig& config) {
    // Build each layer from config
    size_t numLayers = config.layers.size();
    for (int i = 0; i < numLayers; i++) {
        auto layerConfig = config.layers[i];
        Layer* layer = LayerFactory::createLayer(layerConfig,
                                                 input_dim,
                                                 _pDevice,
                                                 _pComputeLibrary,
                                                 i == config.layers.size() - 1);
        dynamicLayers_.push_back(layer);
    }
    dynamicLayers_.back()->setIsTerminal(true);
    
    int first_layer_time_steps = config.first_layer_time_steps > 0 ? config.first_layer_time_steps : 1;
    
    // Build + update input layer
    _pInputLayer->buildBuffers(_pDevice);
    for (int t = 0; t < first_layer_time_steps; ++t) {
        _pInputLayer->updateBufferAt(_pDataManager->getCurrentDataset()->getInputDataAt(t), t);
    }
    
    // Input wiring:
    for (size_t i = 0; i < dynamicLayers_.size(); ++i) {
        dynamicLayers_[i]->connectForwardConnections(i > 0 ? dynamicLayers_[i - 1] : nullptr, _pInputLayer, zeroBuffer_, 0);
    }
    
    // Backward error buffer connections
    for (size_t i = dynamicLayers_.size() - 1; i > 0; --i) {
        dynamicLayers_[i]->connectBackwardConnections(dynamicLayers_[i - 1], _pInputLayer, zeroBuffer_, 0);
    }
}

void NeuralEngine::buildComputePipeline() {
    _pCommandQueue = _pDevice->newCommandQueue();
    NS::Error* pError = nullptr;
    
    _pComputeLibrary = _pDevice->newLibrary(
                                            NS::String::string(multilayerkernels::nnKernelSrc, NS::UTF8StringEncoding),
                                            nullptr, &pError);
    
    if (!_pComputeLibrary) {
        std::cerr << "Error creating compute library: "
        << pError->localizedDescription()->utf8String() << std::endl;
    }
    
    assert(_pComputeLibrary && "Compute library creation failed.");
}

void NeuralEngine::buildBuffers() {
    _pInputLayer->buildBuffers(_pDevice);
    Dataset* currentDataset = _pDataManager->getCurrentDataset();
    for (int t = 0; t < _pInputLayer->getSequenceLength(); ++t) {
        float* inputData = currentDataset->getInputDataAt(t);
        _pInputLayer->updateBufferAt(inputData, t);
    }
    
    if (!zeroBuffer_) {
        zeroBuffer_ = _pDevice->newBuffer(input_dim * sizeof(float), MTL::ResourceStorageModeManaged);
        std::memset(zeroBuffer_->contents(), 0, input_dim * sizeof(float));
    }
    
    areBuffersBuilt = true;
}

void NeuralEngine::computeForward(int batchSize, std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    for (auto layer : dynamicLayers_)
        layer->forward(cmdBuf, batchSize);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        
        
        _pInputLayer->onForwardComplete(_pCommandQueue, batchSize);
        for (auto& layer : dynamicLayers_) {
            layer->onForwardComplete(_pCommandQueue, batchSize);
        }
        
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
#ifdef DEBUG_NETWORK_OUTPUTS
    std::cout << "Output data at timestep 0: " << outputData[0] << ", " << outputData[1] << ", ...\n";
#endif
    
}

void NeuralEngine::computeBackward(int batchSize, std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    
    // Encode backward pass for each layer
    for (auto it = dynamicLayers_.rbegin(); it != dynamicLayers_.rend(); ++it) {
        (*it)->backward(cmdBuf, batchSize);
    }
    
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        
#ifdef DEBUG_NETWORK
        _pInputLayer->debugLog();
        for (auto& layer : dynamicLayers_) {
            layer->debugLog();
        }
#endif
        
        for (auto it = dynamicLayers_.rbegin(); it != dynamicLayers_.rend(); ++it) {
            (*it)->onBackwardComplete(_pCommandQueue, batchSize);
        }
        _pInputLayer->onBackwardComplete(_pCommandQueue, batchSize);
        
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
}

void NeuralEngine::computeForwardBatches(uint32_t totalSamples, int batchesRemaining, std::function<void()> onComplete) {
    
    uint32_t samplesRemaining = mathlib::min<int>((int)ceil(batchesRemaining * batch_size), totalSamples);
    uint32_t currentBatchSize = mathlib::min<int>(batch_size, samplesRemaining);
    std::cout << "⚙️ Forward batches remaining "  << batchesRemaining << " - current batch size " << currentBatchSize << " total samples " << totalSamples << std::endl;

    
    auto samplesProcessed = totalSamples - samplesRemaining;
    
    if (totalSamples == 0 || currentBatchSize < 0) {
        _pLogger->finalizeBatchLoss();
        onComplete();
        
        return;
    }
    
    
    if (currentBatchSize < batch_size) {
        _pLogger->finalizeBatchLoss();
        
        if (currentBatchSize > 0) {
            computeForwardBatches(currentBatchSize, 1, onComplete);
        } else {
            onComplete();
        }
        return;
    }
    
    _pDataManager->loadNextSample();
    
    int terminalSeqLen = dynamicLayers_.back()->getSequenceLength();
    for (int t = 0; t < terminalSeqLen; ++t) {
        float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(t);
        float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(t);
        
        _pInputLayer->updateBufferAt(inBuffer, t);
        dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, t);
    }
    
    computeForward(currentBatchSize, [=, this]() mutable {
        float* predictedData = static_cast<float*>(dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents());
        float* targetData = _pDataManager->getCurrentDataset()->getTargetDataAt(0);
        
        _pLogger->logAnalytics(predictedData, output_dim, targetData, output_dim);

        // When batch is complete, explicitly flush analytics:
        _pLogger->flushAnalytics();
        _pLogger->clearBatchData();
        
        float loss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim);
        _pLogger->accumulateLoss(loss);
        
        samplesProcessed++;
        
        if (samplesProcessed * batch_size % 500 == 0 && samplesProcessed > 0) {
            _pLogger->finalizeBatchLoss();
        }
        
        _pLogger->logAnalytics(predictedData, _pDataManager->getCurrentDataset()->outputDim(), targetData, _pDataManager->getCurrentDataset()->outputDim());
        
        computeForwardBatches(totalSamples, batchesRemaining - 1, onComplete);
    });
}

void NeuralEngine::computeBackwardBatches(uint32_t totalSamples, int batchesRemaining, std::function<void()> onComplete) {
    
    uint32_t samplesRemaining = mathlib::min<int>((int)ceil(batchesRemaining * batch_size), totalSamples);
    uint32_t currentBatchSize = mathlib::min<int>(batch_size, samplesRemaining);
    auto samplesProcessed = totalSamples - samplesRemaining;
    
    std::cout << "⚙️ Backward batches remaining " << batchesRemaining << " - current batch size " << currentBatchSize << std::endl;

    
    if (totalSamples == 0 || currentBatchSize < 0) {
        _pLogger->finalizeBatchLoss();
        onComplete();
        
        return;
    }
    
    
    if (currentBatchSize < batch_size) {
        _pLogger->finalizeBatchLoss();
        
        if (currentBatchSize > 0) {
            computeBackwardBatches(currentBatchSize, 1, onComplete);
        } else {
            onComplete();
        }
        return;
    }
    _pDataManager->loadNextSample();
    
    int terminalSeqLen = dynamicLayers_.back()->getSequenceLength();
    for (int t = 0; t < terminalSeqLen; ++t) {
        float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(t);
        float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(t);
        
        _pInputLayer->updateBufferAt(inBuffer, t);
        dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, t);
    }
    
    computeForward(currentBatchSize, [=, this]() mutable {
        computeBackward(currentBatchSize, [=, this]() mutable {
            float* predictedData = static_cast<float*>(
                                                       dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                                       );
            
            float loss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim);
            _pLogger->accumulateLoss(loss);
            
            globalTimestep++;
            (samplesProcessed)++;
            
            if (samplesProcessed*batch_size % 500 == 0 && samplesProcessed > 0) {
                _pLogger->finalizeBatchLoss();
            }
            
            computeBackwardBatches(totalSamples, batchesRemaining - 1, onComplete);
        });
    });
}

void NeuralEngine::keyPress(KeyPress* kp) {
    _pKeyboardController->keyPress(kp);
}

void NeuralEngine::handleKeyStateChange() {
    _pKeyboardController->handleKeyStateChange();
}


void NeuralEngine::initializeWithDataset(Dataset* dataset) {
    _pDataManager = new DataManager(dataset);
}


void NeuralEngine::saveModel(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    
    size_t layerCount = dynamicLayers_.size();
    file.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
    
    for (Layer* layer : dynamicLayers_) {
        layer->saveParameters(file);
    }
    
    file.close();
    std::cout << "✅ Model parameters saved efficiently (binary) to: " << filepath << std::endl;
}

void NeuralEngine::loadModel(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    
    int layerCount = 0;
    file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
    assert(layerCount == dynamicLayers_.size() && "Layer count mismatch!");
    
    for (Layer* layer : dynamicLayers_) {
        layer->loadParameters(file);
    }
    
    file.close();
    std::cout << "✅ Model parameters loaded efficiently (binary) from: " << filepath << std::endl;
}

void NeuralEngine::shiftBuffers() {
    for (int t = 0; t < _pInputLayer->getSequenceLength() - 1; ++t) {
        memcpy(
               _pInputLayer->getOutputBufferAt(BufferType::Output, t)->contents(),
               _pInputLayer->getOutputBufferAt(BufferType::Output, t + 1)->contents(),
               input_dim * sizeof(float)
               );
        
        memcpy(
               _pDataManager->getCurrentDataset()->getTargetDataAt(t),
               _pDataManager->getCurrentDataset()->getTargetDataAt(t + 1),
               output_dim * sizeof(float)
               );
    }
}
