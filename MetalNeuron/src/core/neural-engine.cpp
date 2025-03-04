#include <iostream>
#include <cassert>
#include <fstream>

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
_pInputLayer(nullptr)
{
    batch_size = config.training.batch_size;
    epochs = config.training.epochs;
    
    input_dim = _pDataManager->inputDim();
    output_dim = _pDataManager->outputDim();
    
    _pLogger = new Logger(outputFileName, config.dataset.type == "function");
    
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        _pLogger->clear();
        TrainingManager::instance().setTraining(false);
        computeForwardIterations(batch_size);
    });
    _pKeyboardController->setLearnCallback([this]() {
        _pLogger->clear();
        TrainingManager::instance().setTraining(true);
        computeBackwardIterations(batch_size);
    });
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _pKeyboardController->setSaveCallback([this]() {
        saveModel("./model.bin");
    });
    
    _pKeyboardController->setLoadCallback([this]() {
        loadModel("./model.bin");
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
    int previousLayerOutputSize = input_dim;
    int first_layer_time_steps = config.first_layer_time_steps > 0 ? config.first_layer_time_steps : 1;
    
    // Build each layer from config
    for (const auto& layerConfig : config.layers) {
        Layer* layer = LayerFactory::createLayer(layerConfig,
                                                 previousLayerOutputSize,
                                                 _pDevice,
                                                 _pComputeLibrary);
        dynamicLayers_.push_back(layer);
    }
    
    // Build + update input layer
    _pInputLayer->buildBuffers(_pDevice);
    for (int t = 0; t < first_layer_time_steps; ++t) {
        _pInputLayer->updateBufferAt(_pDataManager->getCurrentDataset()->getInputDataAt(t), t);
    }
    
    // Input wiring:
    for (size_t i = 0; i < dynamicLayers_.size(); ++i) {
        Layer* prevLayer = (i == 0) ? nullptr : dynamicLayers_[i - 1];
        
        dynamicLayers_[i]->connectInputBuffers(prevLayer, _pInputLayer, zeroBuffer_, 0);
    }
    
    // Backward error buffer connections
    for (size_t i = dynamicLayers_.size() - 1; i > 0; --i) {
        dynamicLayers_[i - 1]->setInputBufferAt(
                                                BufferType::InputErrors,
                                                0,
                                                dynamicLayers_[i]->getOutputBufferAt(BufferType::OutputErrors, 0)
                                                );
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
        zeroBuffer_ = _pDevice->newBuffer(hidden_dim * sizeof(float), MTL::ResourceStorageModeManaged);
        std::memset(zeroBuffer_->contents(), 0, hidden_dim * sizeof(float));
    }
    
    areBuffersBuilt = true;
}

void NeuralEngine::computeForward(std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    for (auto layer : dynamicLayers_)
        layer->forward(cmdBuf);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        
        
        _pInputLayer->onForwardComplete();
        for (auto& layer : dynamicLayers_) {
            layer->onForwardComplete();
        }
        
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
#ifdef DEBUG_NETWORK_OUTPUTS
    std::cout << "Output data at timestep 0: " << outputData[0] << ", " << outputData[1] << ", ...\n";
#endif
    
}

void NeuralEngine::computeBackward(std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    
    // Encode backward pass for each layer
    for (auto it = dynamicLayers_.rbegin(); it != dynamicLayers_.rend(); ++it) {
        (*it)->backward(cmdBuf);
    }
    
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        
        _pInputLayer->onBackwardComplete(_pCommandQueue);
        for (auto& layer : dynamicLayers_) {
            layer->onBackwardComplete(_pCommandQueue);
        }
        
#ifdef DEBUG_NETWORK
        _pInputLayer->debugLog();
        for (auto& layer : dynamicLayers_) {
            layer->debugLog();
        }
#endif
        
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
}

void NeuralEngine::computeBackwardIterations(uint32_t iterations) {
    if (iterations == 0) return;
    
    std::cout << "Backward iterations remaining: " << iterations << std::endl;
    
    shiftBuffers();
    
    _pDataManager->loadNextSample();
    
    int slot = 0;
    
    float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(slot);
    float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(slot);
    
    _pInputLayer->updateBufferAt(inBuffer, slot);
    dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, slot);
    
    
    
    computeForward([this, iterations]() {
        computeBackward([this, iterations]() {
            float* predictedData = static_cast<float*>(
                                                       dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                                       );
            
            float loss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim);
            _pLogger->logLoss(loss);
            
#ifdef DEBUG_GRADIENT_CHECKS
            try {
                std::cout << "Before gradient check, iteration: " << iterations << std::endl;
                GradientChecker checker(this, _pDataManager);
                checker.checkLayerGradients(dynamicLayers_[0]);
                std::cout << "After gradient check, iteration: " << iterations << std::endl;
            } catch (std::exception& error) {
                printf("error: %s\n", error.what());
            }
#endif
            
            globalTimestep++;
            computeBackwardIterations(iterations - 1);
        });
    });
}

void NeuralEngine::keyPress(KeyPress* kp) {
    _pKeyboardController->keyPress(kp);
}

void NeuralEngine::handleKeyStateChange() {
    _pKeyboardController->handleKeyStateChange();
}

void NeuralEngine::runInference() {
    // Implement as needed.
}

void NeuralEngine::computeForwardIterations(uint32_t iterations) {
    if (iterations == 0) return;
    
    std::cout << "Forward iterations remaining: " << iterations << std::endl;
    
    shiftBuffers();
    
    _pDataManager->loadNextSample();
    
    int slot = 0;
    float* inBuffer = _pDataManager->getCurrentDataset()->getInputDataAt(slot);
    float* tgtBuffer = _pDataManager->getCurrentDataset()->getTargetDataAt(slot);
    
    _pInputLayer->updateBufferAt(inBuffer, slot);
    dynamicLayers_.back()->updateTargetBufferAt(tgtBuffer, slot);
    
    computeForward([this, iterations]() {
        float* predictedData = static_cast<float*>(
                                                   dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                                   );
        
        float* targetData =  _pDataManager->getCurrentDataset()->getTargetDataAt(0);
        
        
        float loss = _pDataManager->getCurrentDataset()->calculateLoss(predictedData, output_dim);
        _pLogger->logLoss(loss);
        _pLogger->logAnalytics(predictedData, _pDataManager->getCurrentDataset()->outputDim(), targetData, _pDataManager->getCurrentDataset()->outputDim());
        
#ifdef DEBUG_NETWORK_OUTPUT
        for (int i = 0; i < output_dim; ++i) {
            printf("Output[%d]: %.4f\n", i, predictedData[i]);
        }
#endif
        
        computeForwardIterations(iterations - 1);
    });
}

void NeuralEngine::computeForwardSync() {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    computeForward([semaphore]() {
        dispatch_semaphore_signal(semaphore);
    });
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

void NeuralEngine::computeBackwardSync() {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    computeBackward([semaphore]() {
        dispatch_semaphore_signal(semaphore);
    });
    dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
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
