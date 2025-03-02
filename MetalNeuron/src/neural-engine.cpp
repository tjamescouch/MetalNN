#include "neural-engine.h"
#include "multi-layer-kernels.h"
#include "dropout-layer.h"
#include <iostream>
#include <cassert>
#include <random>

const int num_iterations = 1000;
const int input_dim  = 512;
const int hidden_dim = 512;
const int output_dim = 512;
const char* outputFileName = "multilayer_nn_training.m";


double inputFunc(double index, double timestep) {
    return sin(0.05 * index + 0.1 * timestep);
}

double targetFunc(double index, double timestep) {
    return cos(0.05 * index + 0.1 * timestep);
}

ActivationFunction parseActivation(const std::string& activation) {
    if (activation == "linear") return ActivationFunction::Linear;
    if (activation == "relu") return ActivationFunction::ReLU;
    if (activation == "tanh") return ActivationFunction::Tanh;
    if (activation == "sigmoid") return ActivationFunction::Sigmoid;
    throw std::invalid_argument("Unknown activation: " + activation);
}

std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0, 2*M_PI);

NeuralEngine::NeuralEngine(MTL::Device* pDevice, const ModelConfig& config)
: _pDevice(pDevice->retain()),
areBuffersBuilt(false), currentlyComputing(false), globalTimestep(0), _pDataSourceManager(nullptr), _pInputLayer(nullptr)
{
    
    _pLogger = new Logger(outputFileName);
    
    
    
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        _pLogger->clear();
        computeForwardIterations(num_iterations);
    });
    _pKeyboardController->setLearnCallback([this]() {
        _pLogger->clear();
        computeBackwardIterations(num_iterations);
    });
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    
    buildComputePipeline();
    createDynamicLayers(config);
}

NeuralEngine::~NeuralEngine() {
    for (auto layer : dynamicLayers_)
        delete layer;
    
    delete _pDataSourceManager;
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
    
    int previousLayerOutputSize = input_dim;
    
    int first_layer_time_steps = config.first_layer_time_steps;
    _pInputLayer = new InputLayer(input_dim, first_layer_time_steps);
    
    _pDataSourceManager = new DataSourceManager(input_dim, hidden_dim, output_dim, first_layer_time_steps);
    _pDataSourceManager->initialize([this, config]() {
        _pLogger->clear();
        buildBuffers();
        connectDynamicLayers(config);
    }, inputFunc, targetFunc);
}

void NeuralEngine::connectDynamicLayers(const ModelConfig& config) {
    int previousLayerOutputSize = input_dim;
    int first_layer_time_steps = config.first_layer_time_steps;
    // Build each layer from config
    for (const auto& layerConfig : config.layers) {
        Layer* layer = nullptr;
        if (layerConfig.type == "Dense") {
            int outputSize = layerConfig.params.at("output_size").get_value<int>();
            auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
            ActivationFunction activation = parseActivation(activationStr);
            layer = new DenseLayer(previousLayerOutputSize, outputSize, 1, activation);
            previousLayerOutputSize = outputSize;
        }
        else if (layerConfig.type == "Dropout") {
            float rate = layerConfig.params.at("rate").get_value<float>();
            layer = new DropoutLayer(rate, previousLayerOutputSize, 1);
        }
        else if (layerConfig.type == "BatchNormalization") {
            float epsilon = layerConfig.params.count("epsilon")
            ? layerConfig.params.at("epsilon").get_value<float>()
            : 0.001f;
            layer = new BatchNormalizationLayer(previousLayerOutputSize, 1, epsilon);
        }
        else if (layerConfig.type == "RNN") {
            auto time_steps = layerConfig.time_steps;
            int outputSize = layerConfig.params.at("output_size").get_value<int>();
            auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
            ActivationFunction activation = parseActivation(activationStr);
            layer = new RNNLayer(previousLayerOutputSize, outputSize, time_steps, activation);
            previousLayerOutputSize = outputSize;
        }
        else {
            throw std::invalid_argument("Unsupported layer type");
        }
        
        // Build pipeline & buffers
        layer->buildPipeline(_pDevice, _pComputeLibrary);
        layer->buildBuffers(_pDevice);
        dynamicLayers_.push_back(layer);
    }
    
    // Build + update input layer
    _pInputLayer->buildBuffers(_pDevice);
    for (int t = 0; t < first_layer_time_steps; ++t) {
        _pInputLayer->updateBufferAt(_pDataSourceManager->x, t);
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
    for (int t = 0; t < _pInputLayer->getSequenceLength(); ++t) {
        _pInputLayer->updateBufferAt(_pDataSourceManager->x, t);
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
    
    float* outputData = static_cast<float*>(
                                            dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                            );
    
#ifdef DEBUG_NETWORK_OUTPUTS
    std::cout << "Output data at timestep 0: " << outputData[0] << ", " << outputData[1] << ", ...\n";
#endif
    
    float mse = 0.0f;
    float* targetData = _pDataSourceManager->y.get_data_buffer_at(0);
    int outputDim = dynamicLayers_.back()->outputSize();
    for (int i = 0; i < outputDim; ++i) {
        float diff = targetData[i] - outputData[i];
        mse += diff * diff;
    }
    mse /= outputDim;
    std::printf("Mean Squared Error at timestep 0: %f\n", mse);
}

void NeuralEngine::computeBackward(std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    for (auto it = dynamicLayers_.rbegin(); it != dynamicLayers_.rend(); ++it)
        (*it)->backward(cmdBuf);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        
        _pInputLayer->onBackwardComplete();
        for (auto& layer : dynamicLayers_) {
            layer->onBackwardComplete();
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


void NeuralEngine::shiftBuffers() {
    for (int t = 0; t < _pInputLayer->getSequenceLength() - 1; ++t) {
        memcpy(
               _pInputLayer->getOutputBufferAt(BufferType::Output, t)->contents(),
               _pInputLayer->getOutputBufferAt(BufferType::Output, t + 1)->contents(),
               input_dim * sizeof(float)
               );
        
        memcpy(
               _pDataSourceManager->y.get_data_buffer_at(t),
               _pDataSourceManager->y.get_data_buffer_at(t + 1),
               output_dim * sizeof(float)
               );
    }
}

void NeuralEngine::computeBackwardIterations(uint32_t iterations) {
    if (iterations == 0) return;
    
    std::cout << "Backward iterations remaining: " << iterations << std::endl;
    
    shiftBuffers();
    
    int slot = 0;
    double effectiveTime = distribution(generator);
    
    float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
    float* tgtBuffer = _pDataSourceManager->y.get_data_buffer_at(slot);
    for (int i = 0; i < input_dim; ++i)
        inBuffer[i] = inputFunc(i, effectiveTime);
    for (int i = 0; i < output_dim; ++i)
        tgtBuffer[i] = targetFunc(i, effectiveTime);
    
    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    dynamicLayers_.back()->updateTargetBufferAt(_pDataSourceManager->y, slot);
    
    computeForward([this, iterations]() {
        computeBackward([this, iterations]() {
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
    
    int slot = 0;
    int effectiveTime = globalTimestep;
    
    // Update input data for the new slot.
    float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
    float* tgtBuffer = _pDataSourceManager->y.get_data_buffer_at(slot);
    for (int i = 0; i < input_dim; ++i)
        inBuffer[i] = inputFunc(i, effectiveTime);
    for (int i = 0; i < output_dim; ++i)
        tgtBuffer[i] = targetFunc(i, effectiveTime);
    
    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    dynamicLayers_.back()->updateTargetBufferAt(_pDataSourceManager->y, slot);
    
    
    computeForward([this, iterations]() {
        // Logging outputs clearly to .m file
        std::vector<float*> outputs(1);
        std::vector<float*> targets(1);
        
        outputs[0] = static_cast<float*>(
                                         dynamicLayers_.back()->getOutputBufferAt(BufferType::Output, 0)->contents()
                                         );
        targets[0] = _pDataSourceManager->y.get_data_buffer_at(0);
        
        
        _pLogger->logIteration(*outputs.data(), output_dim, *targets.data(), output_dim);
        
        globalTimestep++;
        computeForwardIterations(iterations - 1);
    });
}
