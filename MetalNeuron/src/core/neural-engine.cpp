#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <unordered_map>

#include "neural-engine.h"
#include "dropout-layer.h"
#include "mnist-dataset.h"
#include "training-manager.h"
#include "math-lib.h"
#include "layer-factory.h"
#include "configuration-manager.h"

NeuralEngine::NeuralEngine(MTL::Device* pDevice, ModelConfig& config, DataManager* pDataManager)
: _pDevice(pDevice->retain()),
areBuffersBuilt(false),
currentlyComputing(false),
_pDataManager(pDataManager),
_pInputLayer(nullptr),
_pLayerFactory(nullptr),
input_dim(0),
output_dim(0),
epochs(0),
zeroBuffer_(nullptr),
filename(config.filename)
{
    batch_size = config.training.batch_size;
    epochs = config.training.epochs;
    
    input_dim = _pDataManager->inputDim();
    output_dim = _pDataManager->outputDim();
    
    int first_layer_time_steps = config.first_layer_time_steps > 0 ? config.first_layer_time_steps : 1;
    _pInputLayer = new InputLayer(input_dim, first_layer_time_steps, batch_size);
    
    _pLayerFactory = new LayerFactory();
    
    Logger::instance().setBatchSize(batch_size);
    Logger::instance().setIsRegression(config.dataset.type == "function");
    
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    
    buildComputePipeline();
    
    createDynamicLayers(config);
}

NeuralEngine::~NeuralEngine() {
    for (auto layer : dynamicLayers_)
        delete layer;
    
    if(_pLayerFactory) delete _pLayerFactory;
    
    if (_pCommandQueue) _pCommandQueue->release();
    if (_pDevice) _pDevice->release();
}

void NeuralEngine::runTraining() {
    Logger::instance().clear();
    TrainingManager::instance().setTraining(true);
    auto pConfig = ConfigurationManager::instance().getConfig();
    
    auto currentEpoch = std::make_shared<int>(0);
    
    // Define epoch callback shared_ptr for recursion
    auto epochCallback = std::make_shared<std::function<void()>>(); //FIXME this is too flowery and complicated for me
    
    *epochCallback = [this, currentEpoch, epochCallback, pConfig]() {
        if (*currentEpoch >= epochs) {
            Logger::log << "✅ Training complete!" << std::endl;
            return;
        }
        
        Logger::log << "🔄 Starting epoch: " << (*currentEpoch + 1) << " / " << epochs << std::endl;
        
        // Run batches for the current epoch, and then call next epoch on completion
        computeBackwardBatches(_pDataManager->getCurrentDataset()->numSamples(), ceil((float)_pDataManager->getCurrentDataset()->numSamples() / pConfig->training.batch_size), [this, currentEpoch, epochCallback]() {
            (*currentEpoch)++;
            (*epochCallback)();
        });
    };
    
    // Start the epoch processing
    (*epochCallback)();
}

void NeuralEngine::runInference() {
    Logger::instance().clear();
    TrainingManager::instance().setTraining(false);
    auto pConfig = ConfigurationManager::instance().getConfig();
    
    computeForwardBatches(_pDataManager->getCurrentDataset()->numSamples(), ceil((float)_pDataManager->getCurrentDataset()->numSamples() / pConfig->training.batch_size), [this]() {
        Logger::log << "✅ Forward pass complete!" << std::endl;
    });
}

void NeuralEngine::saveParameters() {
    this->saveModel(filename + ".bin");
}

void NeuralEngine::loadParameters() {
    loadModel(filename + ".bin");
}

void NeuralEngine::createDynamicLayers(ModelConfig& config) {
    // Clear existing layers
    for (auto layer : dynamicLayers_) {
        delete layer;
    }
    dynamicLayers_.clear();
    
    input_dim = _pDataManager->inputDim();
    output_dim = _pDataManager->outputDim();
    
    
    _pDataManager->initialize(batch_size, [this, &config]() {
        Logger::instance().clear();
        buildBuffers();
        try {
            connectDynamicLayers(config);
        } catch (...) {
            Logger::log << "Caught error connecting layers" << std::endl;
            throw;
        }
    });
}

void NeuralEngine::connectDynamicLayers(ModelConfig& config) {
    // Build each layer from config
    size_t numLayers = config.layers.size();
    for (int i = 0; i < numLayers; i++) {
        auto layerConfig = config.layers[i];
        Layer* layer = _pLayerFactory->createLayer(layerConfig,
                                                 _pDevice,
                                                 _pComputeLibrary,
                                                 i == config.layers.size() - 1);
        dynamicLayers_.push_back(layer);
    }
    dynamicLayers_.back()->setIsTerminal(true);
    
    int first_layer_time_steps = config.first_layer_time_steps > 0 ? config.first_layer_time_steps : 1;
    
    for (int t = 0; t < first_layer_time_steps; ++t) {
        _pInputLayer->updateBufferAt(_pDataManager->getCurrentDataset()->getInputDataAt(t, 0), t);
    }
    
    // Input wiring:
    for (size_t i = 0; i < dynamicLayers_.size(); ++i) {
        dynamicLayers_[i]->connectForwardConnections(i > 0 ? dynamicLayers_[i - 1] : nullptr, _pInputLayer, zeroBuffer_, 0);
    }
    
    // Backward error buffer connections
    for (int64_t i = dynamicLayers_.size() - 1; i >= 0; --i) {
        dynamicLayers_[i]->connectBackwardConnections(i > 0 ? dynamicLayers_[i - 1] : nullptr, _pInputLayer, zeroBuffer_, 0);
    }
}

void NeuralEngine::buildComputePipeline() {
    _pCommandQueue = _pDevice->newCommandQueue();
    NS::Error* pError = nullptr;
    
    _pComputeLibrary = _pDevice->newDefaultLibrary();
    
    if (!_pComputeLibrary) {
        std::cerr << "Error creating compute library: "
        << pError->localizedDescription()->utf8String() << std::endl;
    }
    
    assert(_pComputeLibrary && "Compute library creation failed.");
}

void NeuralEngine::buildBuffers() {
    _pDataManager->loadNextBatch(batch_size);
    
    _pInputLayer->buildBuffers(_pDevice);
    
    
    zeroBuffer_ = _pDevice->newBuffer(input_dim * batch_size * sizeof(float), MTL::ResourceStorageModeManaged);
    std::memset(zeroBuffer_->contents(), 0, input_dim  * batch_size* sizeof(float));
    
    Dataset* currentDataset = _pDataManager->getCurrentDataset();
    float* inputData = currentDataset->getInputDataAt(0, 0);
    _pInputLayer->updateBufferAt(inputData, 0);
    
    areBuffersBuilt = true;
}

void NeuralEngine::computeForward(int batchSize, std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    assert(_pCommandQueue != nullptr && "Command queue not initialized");
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    
    assert(cmdBuf != nullptr && "Command buffer creation failed");
    
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
    if (batchesRemaining <= 0 || totalSamples == 0) {
        Logger::instance().finalizeBatchLoss();
        onComplete();
        return;
    }
    
    uint32_t currentBatchSize = mathlib::min<int>(batch_size, totalSamples);
    
    Logger::log << "⚙️ Forward batches remaining "  << batchesRemaining
    << " - current batch size " << currentBatchSize
    << " total samples remaining " << totalSamples << std::endl;
    
    _pDataManager->loadNextBatch(currentBatchSize);
    
    int terminalSeqLen = 1;
    
    for (int t = 0; t < terminalSeqLen; ++t) {
        float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(t, 0);
        float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(t, 0);
        
        _pInputLayer->updateBufferAt(inBuffer, t, currentBatchSize);
        dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, t, currentBatchSize);
    }
    
    computeForward(currentBatchSize, [=, this]() mutable {
        int t = 0;
        float* predictedData = static_cast<float*>(
                                                   dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, t)->contents()
                                                   );
        
        float* targetData = _pDataManager->getCurrentDataset()->getTargetDataAt(t, 0);
        
        float totalBatchLoss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim * currentBatchSize, targetData);
        
        for (int i = 0; i < currentBatchSize; ++i) {
            Logger::instance().logAnalytics(
                predictedData + i * output_dim, output_dim,
                targetData + i * output_dim, output_dim
            );
        }

        if (currentBatchSize > 0) {
            Logger::instance().accumulateLoss(totalBatchLoss / currentBatchSize, 1);
        }
        assert(!isnan(totalBatchLoss));
        
        
        
        
        if (((totalSamples - currentBatchSize) % 500) == 0) {
            Logger::instance().finalizeBatchLoss();
        }
        
        Logger::instance().flushAnalytics();
        Logger::instance().clearBatchData();
        
        computeForwardBatches(totalSamples - currentBatchSize, batchesRemaining - 1, onComplete);
    });
}


void NeuralEngine::computeBackwardBatches(uint32_t totalSamples, int batchesRemaining, std::function<void()> onComplete) {
    uint32_t samplesRemaining = mathlib::min<int>((int)ceil(batchesRemaining * batch_size), totalSamples);
    uint32_t currentBatchSize = mathlib::min<int>(batch_size, samplesRemaining);
    uint32_t samplesProcessed = totalSamples - samplesRemaining;
    
    Logger::log << "⚙️ Backward batches remaining " << batchesRemaining
    << " - current batch size " << currentBatchSize << std::endl;
    
    if (totalSamples == 0 || currentBatchSize <= 0) {
        Logger::instance().finalizeBatchLoss();
        onComplete();
        return;
    }
    
    _pDataManager->loadNextBatch(currentBatchSize);
    
    int terminalSeqLen = 1;
    for (int t = 0; t < terminalSeqLen; ++t) {
        float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(t, 0);
        float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(t, 0);
        
        _pInputLayer->updateBufferAt(inBuffer, t, batch_size);
        dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, t, batch_size);
    }
    
    computeForward(currentBatchSize, [=, this]() mutable {
        computeBackward(currentBatchSize, [=, this]() mutable {
            
            assert(dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->length() >= batch_size * output_dim * sizeof(float));
            float* predictedData = static_cast<float*>(
                                                       dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                                       );
            
            float* targetData = _pDataManager->getCurrentDataset()->getTargetDataAt(0, 0);
            
            float totalBatchLoss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim * currentBatchSize, targetData);
            //assert(!isnan(batchLoss));
            
            for (int i = 0; i < currentBatchSize; ++i) {
                Logger::instance().logAnalytics(
                    predictedData + i * output_dim, output_dim,
                    targetData + i * output_dim, output_dim
                );
            }
            
            Logger::instance().accumulateLoss(totalBatchLoss / currentBatchSize, 1);
            assert(currentBatchSize > 0);
            //assert(!isnan(batchLoss));
            
            samplesProcessed += currentBatchSize;
            
            if (samplesProcessed % 500 == 0 && samplesProcessed > 0) {
                Logger::instance().finalizeBatchLoss();
            }
            
            Logger::instance().flushAnalytics();
            Logger::instance().clearBatchData();
            
            computeBackwardBatches(totalSamples, batchesRemaining - 1, onComplete);
        });
    });
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
    Logger::log << "✅ Model parameters saved to: " << filepath << std::endl;
}

void NeuralEngine::loadModel(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    
    size_t layerCount = 0;
    file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
    assert(layerCount == dynamicLayers_.size() && "Layer count mismatch!");
    
    for (Layer* layer : dynamicLayers_) {
        layer->loadParameters(file);
    }
    
    file.close();
    Logger::log << "✅ Model parameters loaded from: " << filepath << std::endl;
}

