#include "neural-engine.h"
#include "multi-layer-kernels.h"
#include <iostream>
#include <cassert>

const int input_dim  = 512;
const int hidden_dim = 512;
const int output_dim = 512;
const char* outputFileName = "multilayer_nn_training.m";

double inputFunc(double index, int timestep) {
    return sin(0.05 * index + 0.1 * timestep);
}

double targetFunc(double index, int timestep) {
    return cos(0.05 * index + 0.1 * timestep);
}


NeuralEngine::NeuralEngine(MTL::Device* pDevice, int sequenceLength)
: _pDevice(pDevice->retain()), sequenceLength_(sequenceLength),
  areBuffersBuilt(false), currentlyComputing(false),
  globalTimestep(0)  // initialize the global timestep for animation
{
    _pLogger = new Logger(outputFileName);
    _pDataSourceManager = new DataSourceManager(input_dim, hidden_dim, output_dim, sequenceLength_);
    
    _pDataSourceManager->initialize([this]() {
        _pLogger->clear();
        buildBuffers();
    }, inputFunc, targetFunc);
    
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        _pLogger->clear();
        computeForwardIterations(1000);
    });
    _pKeyboardController->setLearnCallback([this]() {
        computeLearnAndApplyUpdates(1000);
    });
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _semaphore = dispatch_semaphore_create(kMaxFramesInFlight);
    
    // Layers initialization with sequence length
    _pInputLayer = new InputLayer(input_dim, sequenceLength_);
    _pRNNLayer = new RNNLayer(input_dim, hidden_dim, sequenceLength_);
    _pDenseLayer = new DenseLayer(hidden_dim, output_dim, sequenceLength_);
    
    buildComputePipeline();
}

NeuralEngine::~NeuralEngine() {
    delete _pRNNLayer;
    delete _pDenseLayer;
    delete _pInputLayer;
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
    
    assert(_pComputeLibrary && "Compute library creation failed.");
    
    _pRNNLayer->buildPipeline(_pDevice, _pComputeLibrary);
    _pDenseLayer->buildPipeline(_pDevice, _pComputeLibrary);
    _pComputeLibrary->release();
}

void NeuralEngine::buildBuffers() {
    _pInputLayer->buildBuffers(_pDevice);
    _pRNNLayer->buildBuffers(_pDevice);
    _pDenseLayer->buildBuffers(_pDevice);
    
    for (int t = 0; t < sequenceLength_; ++t) {
        _pInputLayer->updateBufferAt(_pDataSourceManager->x, t);
        _pRNNLayer->setInputBufferAt(t, _pInputLayer->getBufferAt(t));
        _pDenseLayer->setInputBufferAt(t, _pRNNLayer->getOutputBufferAt(t));
        _pDenseLayer->updateTargetBufferAt(_pDataSourceManager->y_hat, t);
        _pRNNLayer->setDenseErrorBuffer(_pDenseLayer->getErrorBufferAt(t), t);
    }
    
    
    areBuffersBuilt = true;
}

void NeuralEngine::computeForward(std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    _pRNNLayer->forward(cmdBuf);
    _pDenseLayer->forward(cmdBuf);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    float* outputData = static_cast<float*>(_pDenseLayer->getOutputBufferAt(0)->contents());
    std::cout << "Output data at timestep 0: " << outputData[0] << ", " << outputData[1] << ", ..." << std::endl;
    
}

void NeuralEngine::computeBackward(std::function<void()> onComplete) {
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    _pDenseLayer->backward(cmdBuf);
    _pRNNLayer->backward(cmdBuf);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb) {
        currentlyComputing = false;
        dispatch_semaphore_signal(_semaphore);
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
}

void NeuralEngine::computeLearnAndApplyUpdates(uint32_t iterations) {
    if (iterations == 0) return;
    
    // Shift input and target buffers...
    shiftBuffers();
    // Also shift the RNN hidden state buffers to maintain temporal continuity.
    _pRNNLayer->shiftHiddenStates();
    
    // Use the last slot (always valid) for new data.
    int slot = sequenceLength_ - 1;
    int effectiveTime = globalTimestep + sequenceLength_ - 1;
    
    // Update the input data for the new slot.
    {
        float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
        for (int i = 0; i < input_dim; ++i) {
            inBuffer[i] = static_cast<float>(sin(0.05 * i + 0.1 * effectiveTime));
        }
    }
    // Update the target data for the new slot.
    {
        float* tgtBuffer = _pDataSourceManager->y_hat.get_data_buffer_at(slot);
        for (int i = 0; i < output_dim; ++i) {
            tgtBuffer[i] = static_cast<float>(cos(0.05 * i + 0.1 * effectiveTime));
        }
    }
    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    _pDenseLayer->updateTargetBufferAt(_pDataSourceManager->y_hat, slot);
    
    computeForward([this, iterations]() {
        computeBackward([this, iterations]() {
            std::vector<float*> inputs(sequenceLength_);
            std::vector<float*> hiddenStates(sequenceLength_);
            std::vector<float*> outputs(sequenceLength_);
            std::vector<float*> targets(sequenceLength_);
            std::vector<float*> outputErrors(sequenceLength_);
            std::vector<float*> hiddenErrors(sequenceLength_);
            
            for (int t = 0; t < sequenceLength_; ++t) {
                inputs[t] = static_cast<float*>(_pInputLayer->getBufferAt(t)->contents());
                hiddenStates[t] = static_cast<float*>(_pRNNLayer->getOutputBufferAt(t)->contents());
                outputs[t] = static_cast<float*>(_pDenseLayer->getOutputBufferAt(t)->contents());
                targets[t] = _pDataSourceManager->y_hat.get_data_buffer_at(t);
                outputErrors[t] = static_cast<float*>(_pDenseLayer->getErrorBufferAt(t)->contents());
                hiddenErrors[t] = static_cast<float*>(_pRNNLayer->getErrorBufferAt(t)->contents());
            }
            
            printf("iterations remaining: %d\n", iterations);
            _pLogger->logErrors(outputErrors, output_dim, hiddenErrors, hidden_dim, sequenceLength_);
            
            // Advance the global timestep so that the sinusoid animation advances.
            globalTimestep++;
            computeLearnAndApplyUpdates(iterations - 1);
        });
    });
}


void NeuralEngine::computeForwardIterations(uint32_t iterations) {
    if (iterations == 0) return;
    
    // Shift input and target buffers...
    shiftBuffers();
    // Also shift the RNN hidden state buffers.
    _pRNNLayer->shiftHiddenStates();
    
    int slot = sequenceLength_ - 1;
    int effectiveTime = globalTimestep + sequenceLength_ - 1;
    
    // Update the input data for the new slot.
    {
        float* inBuffer = _pDataSourceManager->x.get_data_buffer_at(slot);
        for (int i = 0; i < input_dim; ++i) {
            inBuffer[i] = static_cast<float>(sin(0.05 * i + 0.1 * effectiveTime));
        }
    }
    // Update the target data for the new slot.
    {
        float* tgtBuffer = _pDataSourceManager->y_hat.get_data_buffer_at(slot);
        for (int i = 0; i < output_dim; ++i) {
            tgtBuffer[i] = static_cast<float>(cos(0.05 * i + 0.1 * effectiveTime));
        }
    }
    _pInputLayer->updateBufferAt(_pDataSourceManager->x, slot);
    _pDenseLayer->updateTargetBufferAt(_pDataSourceManager->y_hat, slot);
    
    computeForward([this, iterations]() {
        std::vector<float*> inputs(sequenceLength_);
        std::vector<float*> hiddenStates(sequenceLength_);
        std::vector<float*> outputs(sequenceLength_);
        std::vector<float*> targets(sequenceLength_);
        
        for (int t = 0; t < sequenceLength_; ++t) {
            inputs[t] = static_cast<float*>(_pInputLayer->getBufferAt(t)->contents());
            hiddenStates[t] = static_cast<float*>(_pRNNLayer->getOutputBufferAt(t)->contents());
            outputs[t] = static_cast<float*>(_pDenseLayer->getOutputBufferAt(t)->contents());
            targets[t] = _pDataSourceManager->y_hat.get_data_buffer_at(t);
        }
        
        printf("iterations remaining: %d\n", iterations);
        _pLogger->logIteration(*outputs.data(), output_dim, *targets.data(), output_dim);
        
        globalTimestep++;
        computeForwardIterations(iterations - 1);
    });
}


void NeuralEngine::keyPress(KeyPress* kp) {
    _pKeyboardController->keyPress(kp);
}

void NeuralEngine::handleKeyStateChange() {
    _pKeyboardController->handleKeyStateChange();
}

void NeuralEngine::shiftBuffers() {
    // Shift input layer buffers
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        memcpy(_pInputLayer->getBufferAt(t)->contents(),
               _pInputLayer->getBufferAt(t + 1)->contents(),
               input_dim * sizeof(float));
    }
    // Shift target data in the DataSource's y_hat buffers
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        memcpy(_pDataSourceManager->y_hat.get_data_buffer_at(t),
               _pDataSourceManager->y_hat.get_data_buffer_at(t + 1),
               output_dim * sizeof(float));
    }
}
