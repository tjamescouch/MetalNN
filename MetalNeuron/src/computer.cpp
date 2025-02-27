/*
 * computer.cpp
 * Created by James Couch on 2025–02–24.
 *
 * This version now uses an RNN–based two–layer network, delegates all
 * DataSource–related functionality to the DataSourceManager class, and
 * delegates keyboard handling to the KeyboardController.
 */

#include <simd/simd.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include "computer.h"
#include "data-source.h"
#include "multi-layer-kernels.h"
#include "keyboard-controller.h"

// stb_image for loading PNG/JPG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Multi-layer dimensions
const int input_dim  = 256;
const int hidden_dim = 256;
const int output_dim = 256;

// Define NUM_ITERATIONS for training as before.
const int NUM_ITERATIONS = 10000;

const char* outputFileName = "multilayer_nn_training.m";

// Example functions for data source initialization.
double inputFunction(double in) {
    return sin(0.050 * in);
}

double expectedOutput(double in) {
    return sin(0.050 * in);
}

#pragma mark – Computer Constructor / Destructor

Computer::Computer(MTL::Device* pDevice)
: _pDataSourceManager(new DataSourceManager(input_dim, hidden_dim, output_dim)),
  _pDevice(pDevice->retain()),
  _pCompileOptions(nullptr),
  areBuffersBuilt(false),
  currentlyComputing(false)
{
    // Build the compute pipeline.
    buildComputePipeline();
    
    // Asynchronously initialize the DataSourceManager.
    _pDataSourceManager->initialize([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            clearOutput();
            buildBuffers();
        });
    }, inputFunction, expectedOutput);
    
    // Create the KeyboardController and set its callbacks.
    _pKeyboardController = new KeyboardController();
    _pKeyboardController->setForwardCallback([this]() {
        this->clearOutput();
        this->computeForwardIterations(NUM_ITERATIONS);
    });
    _pKeyboardController->setLearnCallback([this]() {
        this->computeLearnAndApplyUpdates(NUM_ITERATIONS);
    });
    _pKeyboardController->setClearCallback([this]() {
        this->clearOutput();
    });
    
    _semaphore = dispatch_semaphore_create(Computer::kMaxFramesInFlight);
}

Computer::~Computer()
{
    // Release pipeline states.
    if (_pForwardRnnPipelineState)       _pForwardRnnPipelineState->release();
    if (_pForwardOutputPipelineState)    _pForwardOutputPipelineState->release();
    if (_pLearnOutputPipelineState)      _pLearnOutputPipelineState->release();
    if (_pLearnRnnPipelineState)         _pLearnRnnPipelineState->release();
    
    // Release buffers.
    if (_pBuffer_x)               _pBuffer_x->release();
    if (_pBuffer_hidden)          _pBuffer_hidden->release();
    if (_pBuffer_hidden_prev)     _pBuffer_hidden_prev->release();
    if (_pBuffer_y)               _pBuffer_y->release();
    if (_pBuffer_y_hat)           _pBuffer_y_hat->release();
    
    if (_pBuffer_W1)              _pBuffer_W1->release();
    if (_pBuffer_b1)              _pBuffer_b1->release();
    if (_pBuffer_W2)              _pBuffer_W2->release();
    if (_pBuffer_b2)              _pBuffer_b2->release();
    if (_pBuffer_W_hh)            _pBuffer_W_hh->release();
    
    if (_pBuffer_M1)              _pBuffer_M1->release();
    if (_pBuffer_N1)              _pBuffer_N1->release();
    if (_pBuffer_M2)              _pBuffer_M2->release();
    if (_pBuffer_N2)              _pBuffer_N2->release();
    
    if (_pBuffer_plasticity1)     _pBuffer_plasticity1->release();
    if (_pBuffer_plasticity2)     _pBuffer_plasticity2->release();
    
    if (_pBuffer_age1)            _pBuffer_age1->release();
    if (_pBuffer_age2)            _pBuffer_age2->release();
    
    if (_pBuffer_randomness1)     _pBuffer_randomness1->release();
    if (_pBuffer_randomness2)     _pBuffer_randomness2->release();
    
    if (_pBuffer_error)           _pBuffer_error->release();
    if (_pBuffer_prev_error)      _pBuffer_prev_error->release();
    if (_pBuffer_error_hidden)    _pBuffer_error_hidden->release();
    if (_pBuffer_prev_error_hidden)_pBuffer_prev_error_hidden->release();
    
    if (_pBuffer_WAccumulator1)   _pBuffer_WAccumulator1->release();
    if (_pBuffer_bAccumulator1)   _pBuffer_bAccumulator1->release();
    if (_pBuffer_WAccumulator2)   _pBuffer_WAccumulator2->release();
    if (_pBuffer_bAccumulator2)   _pBuffer_bAccumulator2->release();
    
    // Release function objects.
    if (_pForwardRnnFn)           _pForwardRnnFn->release();
    if (_pForwardOutputFn)        _pForwardOutputFn->release();
    if (_pLearnOutputFn)          _pLearnOutputFn->release();
    if (_pLearnRnnFn)             _pLearnRnnFn->release();
    
    // Release command queue & device.
    if (_pCommandQueue)           _pCommandQueue->release();
    if (_pDevice)                 _pDevice->release();
    
    if (_pDataSourceManager) {
        delete _pDataSourceManager;
        _pDataSourceManager = nullptr;
    }
    
    if (_pKeyboardController) {
        delete _pKeyboardController;
        _pKeyboardController = nullptr;
    }
}

#pragma mark – Compute Pipeline Setup

void Computer::buildComputePipeline()
{
    printf("build compute pipeline (RNN–based multi-layer)\n");
    _pCommandQueue = _pDevice->newCommandQueue();
    
    NS::Error* pError = nullptr;
    _pComputeLibrary = _pDevice->newLibrary(
        NS::String::string(multilayerkernels::nnKernelSrc, NS::UTF8StringEncoding),
        _pCompileOptions,
        &pError
    );
    if (!_pComputeLibrary)
    {
        std::cerr << "Compute library error: "
                  << pError->localizedDescription()->utf8String()
                  << std::endl;
        assert(false);
    }
    
    // Build function objects for updated kernels.
    _pForwardRnnFn = _pComputeLibrary->newFunction(NS::String::string("forward_rnn", NS::UTF8StringEncoding));
    _pForwardOutputFn = _pComputeLibrary->newFunction(NS::String::string("forward_output_layer", NS::UTF8StringEncoding));
    _pLearnOutputFn = _pComputeLibrary->newFunction(NS::String::string("learn_output_layer", NS::UTF8StringEncoding));
    _pLearnRnnFn = _pComputeLibrary->newFunction(NS::String::string("learn_rnn", NS::UTF8StringEncoding));
    
    assert(_pForwardRnnFn);
    assert(_pForwardOutputFn);
    assert(_pLearnOutputFn);
    assert(_pLearnRnnFn);
    
    _pForwardRnnPipelineState = _pDevice->newComputePipelineState(_pForwardRnnFn, &pError);
    _pForwardOutputPipelineState = _pDevice->newComputePipelineState(_pForwardOutputFn, &pError);
    _pLearnOutputPipelineState = _pDevice->newComputePipelineState(_pLearnOutputFn, &pError);
    _pLearnRnnPipelineState = _pDevice->newComputePipelineState(_pLearnRnnFn, &pError);
    
    if (!_pForwardRnnPipelineState ||
        !_pForwardOutputPipelineState ||
        !_pLearnOutputPipelineState ||
        !_pLearnRnnPipelineState) {
        std::cerr << "Error building RNN–based pipeline state." << pError->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    _pComputeLibrary->release();
}

#pragma mark – Buffer Setup

void Computer::buildBuffers()
{
    printf("buildBuffers (RNN–based multi-layer)\n");
    
    // Define dimension values.
    uint m1 = input_dim;    // For input->hidden (W_xh)
    uint n1 = hidden_dim;   // Hidden dimension
    uint m2 = hidden_dim;   // For hidden->output
    uint n2 = output_dim;   // Output dimension
    
    // Buffer for input x.
    _pBuffer_x = _pDevice->newBuffer(_pDataSourceManager->x.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_x->contents(), _pDataSourceManager->x.get_data_buffer(), _pDataSourceManager->x.get_num_data() * sizeof(float));
    _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
    
    // Buffer for hidden activations.
    _pBuffer_hidden = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                          MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_hidden->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_hidden->didModifyRange(NS::Range::Make(0, _pBuffer_hidden->length()));
    
    // Buffer for previous hidden state.
    _pBuffer_hidden_prev = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                               MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_hidden_prev->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_hidden_prev->didModifyRange(NS::Range::Make(0, _pBuffer_hidden_prev->length()));
    
    // Buffer for output activations y.
    _pBuffer_y = _pDevice->newBuffer(output_dim * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_y->contents(), 0, output_dim * sizeof(float));
    _pBuffer_y->didModifyRange(NS::Range::Make(0, _pBuffer_y->length()));
    
    // Buffer for target output y_hat.
    _pBuffer_y_hat = _pDevice->newBuffer(_pDataSourceManager->y_hat.get_num_data() * sizeof(float),
                                         MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_y_hat->contents(), _pDataSourceManager->y_hat.get_data_buffer(), _pDataSourceManager->y_hat.get_num_data() * sizeof(float));
    _pBuffer_y_hat->didModifyRange(NS::Range::Make(0, _pBuffer_y_hat->length()));
    
    // Buffers for recurrent layer weights and biases.
    _pBuffer_W1 = _pDevice->newBuffer(_pDataSourceManager->W1.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W1->contents(), _pDataSourceManager->W1.get_data_buffer(), _pDataSourceManager->W1.get_num_data() * sizeof(float));
    _pBuffer_W1->didModifyRange(NS::Range::Make(0, _pBuffer_W1->length()));
    
    _pBuffer_b1 = _pDevice->newBuffer(_pDataSourceManager->b1.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b1->contents(), _pDataSourceManager->b1.get_data_buffer(), _pDataSourceManager->b1.get_num_data() * sizeof(float));
    _pBuffer_b1->didModifyRange(NS::Range::Make(0, _pBuffer_b1->length()));
    
    // Buffers for output layer weights and biases.
    _pBuffer_W2 = _pDevice->newBuffer(_pDataSourceManager->W2.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W2->contents(), _pDataSourceManager->W2.get_data_buffer(), _pDataSourceManager->W2.get_num_data() * sizeof(float));
    _pBuffer_W2->didModifyRange(NS::Range::Make(0, _pBuffer_W2->length()));
    
    _pBuffer_b2 = _pDevice->newBuffer(_pDataSourceManager->b2.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b2->contents(), _pDataSourceManager->b2.get_data_buffer(), _pDataSourceManager->b2.get_num_data() * sizeof(float));
    _pBuffer_b2->didModifyRange(NS::Range::Make(0, _pBuffer_b2->length()));
    
    // Dimension buffers for RNN hidden layer.
    _pBuffer_M1 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M1->contents(), &m1, sizeof(uint));
    _pBuffer_M1->didModifyRange(NS::Range::Make(0, _pBuffer_M1->length()));
    
    _pBuffer_N1 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N1->contents(), &n1, sizeof(uint));
    _pBuffer_N1->didModifyRange(NS::Range::Make(0, _pBuffer_N1->length()));
    
    // Dimension buffers for output layer.
    _pBuffer_M2 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M2->contents(), &m2, sizeof(uint));
    _pBuffer_M2->didModifyRange(NS::Range::Make(0, _pBuffer_M2->length()));
    
    _pBuffer_N2 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N2->contents(), &n2, sizeof(uint));
    _pBuffer_N2->didModifyRange(NS::Range::Make(0, _pBuffer_N2->length()));
    
    // Allocate buffer for recurrent weights (W_hh) with shape [hidden_dim x hidden_dim].
    _pBuffer_W_hh = _pDevice->newBuffer(hidden_dim * hidden_dim * sizeof(float),
                                        MTL::ResourceStorageModeManaged);
    float* W_hh_data = static_cast<float*>(_pBuffer_W_hh->contents());
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        W_hh_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    _pBuffer_W_hh->didModifyRange(NS::Range::Make(0, _pBuffer_W_hh->length()));
    
    // Error buffers for backpropagation.
    _pBuffer_error = _pDevice->newBuffer(output_dim * sizeof(float),
                                         MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_error->contents(), 0, output_dim * sizeof(float));
    _pBuffer_error->didModifyRange(NS::Range::Make(0, _pBuffer_error->length()));
    
    _pBuffer_error_hidden = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                                MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_error_hidden->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_error_hidden->didModifyRange(NS::Range::Make(0, _pBuffer_error_hidden->length()));
    
    // Accumulator buffers for gradient updates.
    _pBuffer_WAccumulator1 = _pDevice->newBuffer(_pDataSourceManager->W1.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_WAccumulator1->contents(), 0, _pDataSourceManager->W1.get_num_data() * sizeof(float));
    _pBuffer_WAccumulator1->didModifyRange(NS::Range::Make(0, _pBuffer_WAccumulator1->length()));
    
    _pBuffer_bAccumulator1 = _pDevice->newBuffer(_pDataSourceManager->b1.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_bAccumulator1->contents(), 0, _pDataSourceManager->b1.get_num_data() * sizeof(float));
    _pBuffer_bAccumulator1->didModifyRange(NS::Range::Make(0, _pBuffer_bAccumulator1->length()));
    
    _pBuffer_WAccumulator2 = _pDevice->newBuffer(_pDataSourceManager->W2.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_WAccumulator2->contents(), 0, _pDataSourceManager->W2.get_num_data() * sizeof(float));
    _pBuffer_WAccumulator2->didModifyRange(NS::Range::Make(0, _pBuffer_WAccumulator2->length()));
    
    _pBuffer_bAccumulator2 = _pDevice->newBuffer(_pDataSourceManager->b2.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_bAccumulator2->contents(), 0, _pDataSourceManager->b2.get_num_data() * sizeof(float));
    _pBuffer_bAccumulator2->didModifyRange(NS::Range::Make(0, _pBuffer_bAccumulator2->length()));
    
    areBuffersBuilt = true;
}

#pragma mark – Forward Pass

// The forward pass now consists of two kernel dispatches:
//  1. RNN Layer: Compute hidden activations using forward_rnn.
//  2. Output Layer: Compute output activations using forward_output_layer.
// After the forward pass, the computed hidden state is copied to the hidden_prev buffer.
void Computer::computeForward(std::function<void()> onComplete)
{
    printf("computeForward (RNN–based multi-layer)\n");
    
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Performing forward pass..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    // --- Layer 1: RNN Layer (Input -> Hidden) ---
    {
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(_pForwardRnnPipelineState);
        enc->setBuffer(_pBuffer_x, 0, 0);
        enc->setBuffer(_pBuffer_hidden_prev, 0, 1);
        enc->setBuffer(_pBuffer_hidden, 0, 2);
        enc->setBuffer(_pBuffer_W1, 0, 3);
        enc->setBuffer(_pBuffer_W_hh, 0, 4);
        enc->setBuffer(_pBuffer_b1, 0, 5);
        enc->setBuffer(_pBuffer_M1, 0, 6); // input_dim
        enc->setBuffer(_pBuffer_N1, 0, 7); // hidden_dim
        uint32_t threads = hidden_dim;
        MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
        MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
        enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        enc->endEncoding();
    }
    
    // --- Layer 2: Output Layer (Hidden -> Output) ---
    {
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(_pForwardOutputPipelineState);
        enc->setBuffer(_pBuffer_hidden, 0, 0);
        enc->setBuffer(_pBuffer_y, 0, 1);
        enc->setBuffer(_pBuffer_W2, 0, 2);
        enc->setBuffer(_pBuffer_b2, 0, 3);
        enc->setBuffer(_pBuffer_N1, 0, 4); // hidden_dim
        enc->setBuffer(_pBuffer_N2, 0, 5); // output_dim
        uint32_t threads = output_dim;
        MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
        MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
        enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        enc->endEncoding();
    }
    
    // --- Copy Hidden State to Hidden_prev for next iteration ---
    {
        MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
        blitEncoder->copyFromBuffer(_pBuffer_hidden, 0, _pBuffer_hidden_prev, 0, _pBuffer_hidden->length());
        blitEncoder->synchronizeResource(_pBuffer_y);
        blitEncoder->endEncoding();
    }
    
    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
        std::cout << "Forward pass complete." << std::endl;
        currentlyComputing = false;
        onComplete();
    });
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    pPool->release();
}

#pragma mark – Learning (Backpropagation) & Apply Updates

// This method performs one iteration of backpropagation and then applies weight updates.
void Computer::computeLearn(std::function<void()> onComplete)
{
    this->computeForward([this, onComplete](){
        printf("computeLearn (RNN–based multi-layer)\n");
        
        if (!areBuffersBuilt) return;
        if (currentlyComputing) return;
        
        currentlyComputing = true;
        std::cout << "Performing learning (backpropagation)..." << std::endl;
        
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        
        MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
        assert(cmdBuf);
        
        // --- Output Layer Learning ---
        {
            MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(_pLearnOutputPipelineState);
            enc->setBuffer(_pBuffer_hidden, 0, 0);
            enc->setBuffer(_pBuffer_W2, 0, 1);
            enc->setBuffer(_pBuffer_b2, 0, 2);
            enc->setBuffer(_pBuffer_y, 0, 3);
            enc->setBuffer(_pBuffer_y_hat, 0, 4);
            enc->setBuffer(_pBuffer_error, 0, 5);
            enc->setBuffer(_pBuffer_N1, 0, 6); // hidden_dim
            enc->setBuffer(_pBuffer_N2, 0, 7); // output_dim
            uint32_t threads = output_dim;
            MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
            MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
            enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
            enc->endEncoding();
        }
        
        // --- RNN Layer Learning ---
        {
            MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
            enc->setComputePipelineState(_pLearnRnnPipelineState);
            enc->setBuffer(_pBuffer_x, 0, 0);
            enc->setBuffer(_pBuffer_hidden_prev, 0, 1);
            enc->setBuffer(_pBuffer_W1, 0, 2);
            enc->setBuffer(_pBuffer_W_hh, 0, 3);
            enc->setBuffer(_pBuffer_b1, 0, 4);
            enc->setBuffer(_pBuffer_hidden, 0, 5);
            enc->setBuffer(_pBuffer_error_hidden, 0, 6);
            enc->setBuffer(_pBuffer_M1, 0, 7); // input_dim
            enc->setBuffer(_pBuffer_N1, 0, 8); // hidden_dim
            uint32_t threads = hidden_dim;
            MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
            MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
            enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
            enc->endEncoding();
        }
        
        cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
            dispatch_semaphore_signal(this->_semaphore);
            std::cout << "Learning complete." << std::endl;
            currentlyComputing = false;
            logError();
            onComplete();
        });
        
        cmdBuf->commit();
        dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
        pPool->release();
    });
}

void Computer::computeLearnAndApplyUpdates(uint32_t iterations)
{
    printf("computeLearnAndApplyUpdates (RNN–based multi-layer) - iterations remaining = %d\n", (int)iterations);
    this->computeLearn([this, iterations]() {
        // Update input data for the next iteration.
        _pDataSourceManager->x.build([iterations](double x){ return inputFunction(x - iterations); });
        _pDataSourceManager->y_hat.build([iterations](double x){ return inputFunction(x - iterations); });
        
        std::memcpy(_pBuffer_x->contents(), _pDataSourceManager->x.get_data_buffer(), _pDataSourceManager->x.get_num_data() * sizeof(float));
        _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
        
        std::memcpy(_pBuffer_y_hat->contents(), _pDataSourceManager->y_hat.get_data_buffer(), _pDataSourceManager->y_hat.get_num_data() * sizeof(float));
        _pBuffer_y_hat->didModifyRange(NS::Range::Make(0, _pBuffer_y_hat->length()));
        
        if (iterations > 0) {
            this->computeLearnAndApplyUpdates(iterations - 1);
        }
    });
}

void Computer::computeForwardIterations(uint32_t iterations)
{
    printf("computeForwardIterations (RNN–based multi-layer)\n");
    this->computeForward([this, iterations]() {
        printf("Forward Iterations remaining=%d\n", iterations);
        _pDataSourceManager->x.build([iterations](double x){ return inputFunction(x - iterations); });
        std::memcpy(_pBuffer_x->contents(), _pDataSourceManager->x.get_data_buffer(), _pDataSourceManager->x.get_num_data() * sizeof(float));
        _pBuffer_x->didModifyRange(NS::Range(0, _pBuffer_x->length()));
        _pBuffer_hidden->didModifyRange(NS::Range(0, _pBuffer_hidden->length()));
        _pBuffer_y->didModifyRange(NS::Range(0, _pBuffer_y->length()));
        _pBuffer_error->didModifyRange(NS::Range(0, _pBuffer_error->length()));
        _pBuffer_error_hidden->didModifyRange(NS::Range(0, _pBuffer_error_hidden->length()));
        
        this->extractAllResults(iterations);
        
        if (iterations > 0) {
            this->computeForwardIterations(iterations - 1);
        }
    });
}

#pragma mark – Logging & Output

void Computer::logError()
{
    float* error = static_cast<float*>(_pBuffer_error->contents());
    
    float avg_error = 0.0f;
    for (int i = 0; i < input_dim; i++) {
        avg_error += error[i];
    }
    avg_error /= input_dim;
    printf("AVG OUTPUT ERROR: %f\n", abs(avg_error));
    
    float* error_hidden = static_cast<float*>(_pBuffer_error_hidden->contents());
    
    float avg_error_hidden = 0.0f;
    for (int i = 0; i < input_dim; i++) {
        avg_error_hidden += error_hidden[i];
    }
    avg_error_hidden /= input_dim;
    printf("AVG HIDDEN ERROR: %f\n", abs(avg_error_hidden));
}

void Computer::logInformation(const std::string& filename, int remainingIterations)
{
    printf("logInformation\n");
    std::ofstream logFile(filename, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file!" << std::endl;
        return;
    }
    
    logFile << "clf; hold on;" << std::endl;
    logFile << "ylim([-1 1]);" << std::endl;
    
    float* inputPtr = static_cast<float*>(_pBuffer_x->contents());
    float* hiddenPtr = static_cast<float*>(_pBuffer_hidden->contents());
    float* outputPtr = static_cast<float*>(_pBuffer_y->contents());
    float* targetPtr = static_cast<float*>(_pBuffer_y_hat->contents());
    
    uint64_t length = _pBuffer_y->length() / sizeof(float);
    
    logFile << "# Logging iteration" << std::endl;
    logFile << "x = [ ";
    for (uint64_t i = 0; i < length; i++) {
        if (i != 0)
            logFile << ", ";
        logFile << i;
    }
    logFile << " ]" << std::endl;
    
    logFile << "input = [ ";
    for (uint64_t i = 0; i < length; i++) {
        if (i != 0)
            logFile << ", ";
        logFile << inputPtr[i];
    }
    logFile << " ]" << std::endl;
    
    logFile << "hidden = [ ";
    for (uint64_t i = 0; i < hidden_dim; i++) {
        if (i != 0)
            logFile << ", ";
        logFile << hiddenPtr[i];
    }
    logFile << " ]" << std::endl;
    
    logFile << "output = [ ";
    for (uint64_t i = 0; i < length; i++) {
        if (i != 0)
            logFile << ", ";
        logFile << outputPtr[i];
    }
    logFile << " ]" << std::endl;
    
    logFile << "target = [ ";
    for (uint64_t i = 0; i < length; i++) {
        if (i != 0)
            logFile << ", ";
        logFile << targetPtr[i];
    }
    logFile << " ]" << std::endl;
    
    logFile << "scatter(1:length(input), input, 'b');" << std::endl;
    logFile << "scatter(1:length(output), output, 'r');" << std::endl;
    logFile << "hold off; pause(0.01);" << std::endl;
    
    logFile.close();
}

void Computer::extractAllResults(int remainingIterations)
{
    printf("extractAllResults\n");
    logInformation(outputFileName, remainingIterations);
}

void Computer::clearOutput()
{
    printf("clearOutput\n");
    std::ofstream logFile(outputFileName, std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file!" << std::endl;
        return;
    }
    logFile << std::endl;
}

#pragma mark – User Input Handling

void Computer::keyPress(KeyPress* kp)
{
    _pKeyboardController->keyPress(kp);
}

void Computer::handleKeyStateChange()
{
    _pKeyboardController->handleKeyStateChange();
}
