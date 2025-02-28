#include <cmath>
#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include "neural-engine.h"
#include "data-source-manager.h"
#include "keyboard-controller.h"
#include "logger.h"
#include "rnn-layer.h"
#include "dense-layer.h"
#include "multi-layer-kernels.h"



// Define dimensions and training iterations.
const int input_dim  = 256;
const int hidden_dim = 256;
const int output_dim = 256;
const int NUM_ITERATIONS = 10000;
const char* outputFileName = "multilayer_nn_training.m";

// Example functions for data source initialization.
double inputFunction(double in) {
    return sin(0.050 * in);
}
double expectedOutput(double in) {
    return sin(0.050 * in);
}

NeuralEngine::NeuralEngine(MTL::Device* pDevice)
: _pDataSourceManager(new DataSourceManager(input_dim, hidden_dim, output_dim)),
_pDevice(pDevice->retain()),
_pCompileOptions(nullptr),
areBuffersBuilt(false),
currentlyComputing(false),
_pInputLayer(nullptr),
_pRNNLayer(nullptr),
_pDenseLayer(nullptr)
{
    _pLogger = new Logger(outputFileName);
    
    _pDataSourceManager->initialize([this]() {
        _pLogger->clear();
        buildBuffers();
    }, inputFunction, expectedOutput);
    
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        _pLogger->clear();
        computeForwardIterations(NUM_ITERATIONS);
    });
    _pKeyboardController->setLearnCallback([this]() {
        computeLearnAndApplyUpdates(NUM_ITERATIONS);
    });
    _pKeyboardController->setClearCallback([this]() {
        _pLogger->clear();
    });
    
    _semaphore = dispatch_semaphore_create(NeuralEngine::kMaxFramesInFlight);
    
    // Compose the network from layers.
    _pInputLayer = new InputLayer(input_dim);
    _pRNNLayer = new RNNLayer(input_dim, hidden_dim);
    _pDenseLayer = new DenseLayer(hidden_dim, output_dim);
    
    buildComputePipeline();
}


NeuralEngine::~NeuralEngine() {
    if (_pRNNLayer) {
        delete _pRNNLayer;
        _pRNNLayer = nullptr;
    }
    if (_pDenseLayer) {
        delete _pDenseLayer;
        _pDenseLayer = nullptr;
    }
    if (_pDataSourceManager) {
        delete _pDataSourceManager;
        _pDataSourceManager = nullptr;
    }
    if (_pKeyboardController) {
        delete _pKeyboardController;
        _pKeyboardController = nullptr;
    }
    if (_pLogger) {
        delete _pLogger;
        _pLogger = nullptr;
    }
    if (_pComputeLibrary) _pComputeLibrary->release();
    if (_pCommandQueue) _pCommandQueue->release();
    if (_pDevice) _pDevice->release();
}

void NeuralEngine::buildComputePipeline() {
    std::cout << "Building compute pipeline (NeuralEngine)" << std::endl;
    _pCommandQueue = _pDevice->newCommandQueue();
    
    NS::Error* pError = nullptr;
    _pComputeLibrary = _pDevice->newLibrary(
                                            NS::String::string(multilayerkernels::nnKernelSrc, NS::UTF8StringEncoding),
                                            _pCompileOptions,
                                            &pError
                                            );
    if (!_pComputeLibrary) {
        std::cerr << "Compute library error: " << pError->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    // Let each layer build its own pipeline.
    _pRNNLayer->buildPipeline(_pDevice, _pComputeLibrary);
    _pDenseLayer->buildPipeline(_pDevice, _pComputeLibrary);
    
    _pComputeLibrary->release();
}

void NeuralEngine::buildBuffers() {
    std::cout << "Building buffers (NeuralEngine)" << std::endl;
    
    // Build DenseLayer buffers first.
    _pDenseLayer->buildBuffers(_pDevice);
    
    // Now build RNNLayer buffers.
    _pRNNLayer->buildBuffers(_pDevice);
    
    // Set the DenseLayer's error buffer for the RNNLayer.
    _pRNNLayer->setDenseErrorBuffer(_pDenseLayer->getErrorBuffer());
    
    // Build input buffers.
    _pInputLayer->buildBuffers(_pDevice);
    _pInputLayer->updateBuffer(_pDataSourceManager->x);
    _pRNNLayer->setInputBuffer(_pInputLayer->getBuffer());
    
    // Chain: set DenseLayer's input to be RNNLayer's output.
    _pDenseLayer->setInputBuffer(_pRNNLayer->getOutputBuffer());
    
    areBuffersBuilt = true;
}


void NeuralEngine::computeForward(std::function<void()> onComplete) {
    std::cout << "Performing forward pass (NeuralEngine)" << std::endl;
    if (!areBuffersBuilt || currentlyComputing) return;
    currentlyComputing = true;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    auto cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    _pRNNLayer->forward(cmdBuf);
    _pDenseLayer->forward(cmdBuf);
    
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(_semaphore);
        currentlyComputing = false;
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    pPool->release();
}

void NeuralEngine::computeLearn(std::function<void()> onComplete) {
    computeForward([this, onComplete](){
        std::cout << "Performing learning pass (NeuralEngine)" << std::endl;
        if (!areBuffersBuilt || currentlyComputing) return;
        currentlyComputing = true;
        
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        auto cmdBuf = _pCommandQueue->commandBuffer();
        assert(cmdBuf);
        
        _pDenseLayer->backward(cmdBuf);
        _pRNNLayer->backward(cmdBuf);
        
        cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
            dispatch_semaphore_signal(_semaphore);
            currentlyComputing = false;
            // (For demonstration, we could log errors here.)
            onComplete();
        });
        cmdBuf->commit();
        dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
        pPool->release();
    });
}

void NeuralEngine::computeLearnAndApplyUpdates(uint32_t iterations) {
    std::cout << "computeLearnAndApplyUpdates, iterations remaining = " << iterations << std::endl;
    computeLearn([this, iterations]() {
        _pDataSourceManager->x.build([iterations](double x){ return inputFunction(x - iterations); });
        _pDataSourceManager->y_hat.build([iterations](double x){ return inputFunction(x - iterations); });
        
        _pInputLayer->updateBuffer(_pDataSourceManager->x);
        
        // Retrieve error data from the learning passes.
        float* outputError = static_cast<float*>(_pDenseLayer->getErrorBuffer()->contents());
        long outputErrorCount = _pDenseLayer->getErrorBuffer()->length() / sizeof(float);
        
        float* hiddenError = static_cast<float*>(_pRNNLayer->getErrorBuffer()->contents());
        long hiddenErrorCount = _pRNNLayer->getErrorBuffer()->length() / sizeof(float);
        
        // Log errors.
        _pLogger->logErrors(outputError, (int)outputErrorCount,
                            hiddenError, (int)hiddenErrorCount);
        
        if (iterations > 0) {
            computeLearnAndApplyUpdates(iterations - 1);
        }
    });
}

void NeuralEngine::computeForwardIterations(uint32_t iterations) {
    std::cout << "computeForwardIterations, iterations remaining = " << iterations << std::endl;
    computeForward([this, iterations]() {
        _pDataSourceManager->x.build([iterations](double x){ return inputFunction(x - iterations); });
        _pInputLayer->updateBuffer(_pDataSourceManager->x);
        
        // Retrieve iteration information.
        float* inputData = static_cast<float*>(_pInputLayer->getBuffer()->contents());
        long inputCount = _pInputLayer->getBuffer()->length() / sizeof(float);
        
        float* hiddenData = static_cast<float*>(_pRNNLayer->getOutputBuffer()->contents());
        long hiddenCount = _pRNNLayer->getOutputBuffer()->length() / sizeof(float);
        
        float* outputData = static_cast<float*>(_pDenseLayer->getOutputBuffer()->contents());
        long outputCount = _pDenseLayer->getOutputBuffer()->length() / sizeof(float);
        
        // Assume DataSource y_hat has getData() and getSize() methods.
        float* targetData = static_cast<float*>(_pDataSourceManager->y_hat.get_data_buffer());
        long targetCount = _pDataSourceManager->y_hat.get_num_data() / sizeof(float);
        
        // Log iteration information.
        _pLogger->logIteration(inputData,  (int)inputCount,
                               hiddenData, (int)hiddenCount,
                               outputData, (int)outputCount,
                               targetData, (int)targetCount);
        
        if (iterations > 0) {
            computeForwardIterations(iterations - 1);
        }
    });
}

void NeuralEngine::keyPress(KeyPress* kp) {
    _pKeyboardController->keyPress(kp);
}

void NeuralEngine::handleKeyStateChange() {
    _pKeyboardController->handleKeyStateChange();
}
