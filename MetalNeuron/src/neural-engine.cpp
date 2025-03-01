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

//FIXME - find a better place for this
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
: _pDevice(pDevice->retain()), sequenceLength_(0),
  areBuffersBuilt(false), currentlyComputing(false), globalTimestep(0)
{
    sequenceLength_ = config.time_steps;
    
    _pLogger = new Logger(outputFileName);
    _pDataSourceManager = new DataSourceManager(input_dim, hidden_dim, output_dim, sequenceLength_);
    _pInputLayer = new InputLayer(input_dim, sequenceLength_);
    
    _pDataSourceManager->initialize([this, config]() {
        _pLogger->clear();
        buildBuffers();
        createDynamicLayers(config);
    }, inputFunc, targetFunc);
    
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        _pLogger->clear();
        computeForwardIterations(num_iterations);
    });
    _pKeyboardController->setLearnCallback([this]() {
        _pLogger->clear();
        computeLearnAndApplyUpdates(num_iterations);
    });
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    
    buildComputePipeline();
}

void NeuralEngine::createDynamicLayers(const ModelConfig& config) {
    if (!dynamicLayers_.empty()) {
        for (auto layer : dynamicLayers_)
            delete layer;
        dynamicLayers_.clear();
    }

    int previousLayerOutputSize = input_dim;

    for (const auto& layerConfig : config.layers) {
        if (layerConfig.type == "Dense") {
            int outputSize = layerConfig.params.at("output_size").get_value<int>();
            auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
            ActivationFunction activation = parseActivation(activationStr);
            auto dense = new DenseLayer(previousLayerOutputSize, outputSize, sequenceLength_, activation);
            
            dense->buildPipeline(_pDevice, _pComputeLibrary);
            dense->buildBuffers(_pDevice);
            dynamicLayers_.push_back(dense);
            
            previousLayerOutputSize = outputSize;
        } else if (layerConfig.type == "Dropout") {
            float rate = layerConfig.params.at("rate").get_value<float>();
            auto dropout = new DropoutLayer(rate, previousLayerOutputSize, sequenceLength_);
            
            dropout->buildPipeline(_pDevice, _pComputeLibrary);
            dropout->buildBuffers(_pDevice);
            assert(dropout->getOutputBufferAt(0) && "Dropout output buffer at timestep 0 is null after initialization!");
            dynamicLayers_.push_back(dropout);
        } else if (layerConfig.type == "BatchNormalization") {
            float epsilon = 0.001f;  // default epsilon
            if (layerConfig.params.count("epsilon")) {
                epsilon = layerConfig.params.at("epsilon").get_value<float>();
            }

            auto batchNorm = new BatchNormalizationLayer(previousLayerOutputSize, sequenceLength_, epsilon);

            batchNorm->buildPipeline(_pDevice, _pComputeLibrary);
            batchNorm->buildBuffers(_pDevice);

            dynamicLayers_.push_back(batchNorm);
        } else if (layerConfig.type == "RNN") {
            int outputSize = layerConfig.params.at("output_size").get_value<int>();
            auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
            ActivationFunction activation = parseActivation(activationStr);
            auto rnn = new RNNLayer(previousLayerOutputSize, outputSize, sequenceLength_, activation);

            rnn->buildPipeline(_pDevice, _pComputeLibrary);
            rnn->buildBuffers(_pDevice);
            dynamicLayers_.push_back(rnn);

            previousLayerOutputSize = outputSize;
        } else {
            throw new std::invalid_argument("Unsupported layer type");
        }
    }

    // ⚙️ Connect input layer buffers
    _pInputLayer->buildBuffers(_pDevice);
    for (int t = 0; t < sequenceLength_; ++t) {
        _pInputLayer->updateBufferAt(_pDataSourceManager->x, t);
        dynamicLayers_.front()->setInputBufferAt(t, _pInputLayer->getOutputBufferAt(t));
    }

    // ⚙️ Connect subsequent layers
    for (size_t i = 1; i < dynamicLayers_.size(); ++i) {
        for (int t = 0; t < sequenceLength_; ++t) {
            dynamicLayers_[i]->setInputBufferAt(t, dynamicLayers_[i-1]->getOutputBufferAt(t));
        }
    }
    
    // ⚙️ Backward connections (this is where you use setOutputErrorBufferAt/getInputErrorBufferAt):
    for (size_t i = dynamicLayers_.size() - 1; i > 0; --i) {
        for (int t = 0; t < sequenceLength_; ++t) {
            dynamicLayers_[i-1]->setOutputErrorBufferAt(t, dynamicLayers_[i]->getInputErrorBufferAt(t));
        }
    }
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
    for (int t = 0; t < sequenceLength_; ++t) {
        _pInputLayer->updateBufferAt(_pDataSourceManager->x, t);
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
        onComplete();
    });

    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);

    float* outputData = static_cast<float*>(dynamicLayers_.back()->getOutputBufferAt(0)->contents());

#ifdef DEBUG_NETWORK
    std::cout << "Output data at timestep 0: " << outputData[0] << ", " << outputData[1] << ", ...\n";
#endif

    float mse = 0.0f;
    float* targetData = _pDataSourceManager->y_hat.get_data_buffer_at(0);
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
        
        for (int t = 0; t < sequenceLength_; ++t) {
            auto layer = dynamicLayers_.back(); // Typically check last layer first
            float* gradients = static_cast<float*>(layer->getErrorBufferAt(t)->contents());
            printf("Gradients sample timestep %d: %f, %f, %f\n", t, gradients[0], gradients[1], gradients[2]);
        }
        
        onComplete();
    });

    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
}

void NeuralEngine::shiftBuffers() {
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        memcpy(_pInputLayer->getOutputBufferAt(t)->contents(),
               _pInputLayer->getOutputBufferAt(t + 1)->contents(),
               input_dim * sizeof(float));
        memcpy(_pDataSourceManager->y_hat.get_data_buffer_at(t),
               _pDataSourceManager->y_hat.get_data_buffer_at(t + 1),
               output_dim * sizeof(float));
    }
}

void NeuralEngine::computeLearnAndApplyUpdates(uint32_t iterations) {
    if (iterations == 0) return;

    shiftBuffers();

    int slot = sequenceLength_ - 1;
    double effectiveTime = distribution(generator);

    float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
    float* tgtBuffer = _pDataSourceManager->y_hat.get_data_buffer_at(slot);
    for (int i = 0; i < input_dim; ++i)
        inBuffer[i] = inputFunc(i, effectiveTime);
    for (int i = 0; i < output_dim; ++i)
        tgtBuffer[i] = targetFunc(i, effectiveTime);

    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    dynamicLayers_.back()->updateTargetBufferAt(_pDataSourceManager->y_hat, slot);

    computeForward([this, iterations]() {
        computeBackward([this, iterations]() {
            globalTimestep++;
            computeLearnAndApplyUpdates(iterations - 1);
#ifdef DEBUG_NETWORK
            float* errData = static_cast<float*>(dynamicLayers_.back()->getErrorBufferAt(0)->contents());
            std::cout << "Error check after backward: "
                      << errData[0] << ", " << errData[1] << ", ..." << std::endl;
#endif
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

    shiftBuffers();

    int slot = sequenceLength_ - 1;
    int effectiveTime = globalTimestep + sequenceLength_ - 1;

    // Update input data for the new slot.
    float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
    float* tgtBuffer = _pDataSourceManager->y_hat.get_data_buffer_at(slot);
    for (int i = 0; i < input_dim; ++i)
        inBuffer[i] = inputFunc(i, effectiveTime);
    for (int i = 0; i < output_dim; ++i)
        tgtBuffer[i] = targetFunc(i, effectiveTime);

    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    dynamicLayers_.back()->updateTargetBufferAt(_pDataSourceManager->y_hat, slot);
    

    computeForward([this, iterations]() {
        // Logging outputs clearly to .m file
        std::vector<float*> outputs(sequenceLength_);
        std::vector<float*> targets(sequenceLength_);

        for (int t = 0; t < sequenceLength_; ++t) {
            outputs[t] = static_cast<float*>(dynamicLayers_.back()->getOutputBufferAt(t)->contents());
            targets[t] = _pDataSourceManager->y_hat.get_data_buffer_at(t);
        }

        _pLogger->logIteration(*outputs.data(), output_dim, *targets.data(), output_dim);

        globalTimestep++;
        computeForwardIterations(iterations - 1);
    });
}
