/*
 * computer.cpp
 * Created by James Couch on 2025-02-24.
 *
 * This version extends the original single–layer implementation to support
 * a two–layer network:
 *   - Layer 1: Input (dimension: input_dim) → Hidden (hidden_dim neurons)
 *   - Layer 2: Hidden (hidden_dim) → Output (output_dim neurons)
 *
 * It preserves original functionality such as asynchronous initialization,
 * key bindings (F for forward, L for learn, C for clear), logging, etc.
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

// stb_image for loading PNG/JPG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Multi-layer dimensions (adjust as needed)
const int input_dim  = 256;
const int hidden_dim = 256;
const int output_dim = 256;

const char* outputFileName = "multilayer_nn_training.m";
const int NUM_ITERATIONS = 10000;

// Example functions for data source initialization.
double inputFunction(double in) {
    return sin(0.050 * in);// > 0 ? 1.0f : -1.0f;
}

double expectedOutput(double in) {
    return sin(0.050 * in);
}

#pragma mark - Computer Constructor / Destructor

Computer::Computer(MTL::Device* pDevice)
: x(input_dim, 1),
y_hat(output_dim, 1),
W1(input_dim, hidden_dim),
b1(hidden_dim, 1),
W2(hidden_dim, output_dim),
b2(output_dim, 1),
rand1(hidden_dim, 1),
rand2(output_dim, 1),
_pDevice(pDevice->retain()),
_pCompileOptions(nullptr),
areBuffersBuilt(false),
currentlyComputing(false)
{
    buildComputePipeline();
    
    // Initialize data sources asynchronously.
    rand1.initRandomAsync([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            rand2.initRandomAsync([this]() {
                dispatch_async(dispatch_get_main_queue(), ^{
                    x.buildAsync(inputFunction, [this]() {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            y_hat.buildAsync(expectedOutput, [this]() {
                                dispatch_async(dispatch_get_main_queue(), ^{
                                    W1.initRandomAsync([this]() {
                                        dispatch_async(dispatch_get_main_queue(), ^{
                                            b1.initRandomAsync([this]() {
                                                dispatch_async(dispatch_get_main_queue(), ^{
                                                    W2.initRandomAsync([this]() {
                                                        dispatch_async(dispatch_get_main_queue(), ^{
                                                            b2.initRandomAsync([this]() {
                                                                dispatch_async(dispatch_get_main_queue(), ^{
                                                                    clearOutput();
                                                                    buildBuffers();
                                                                });
                                                            });
                                                        });
                                                    });
                                                });
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
    
    _semaphore = dispatch_semaphore_create(Computer::kMaxFramesInFlight);
}

Computer::~Computer()
{
    // Release pipeline states.
    if (_pForwardLayerPipelineState)      _pForwardLayerPipelineState->release();
    if (_pForwardLayerPipelineState2)     _pForwardLayerPipelineState2->release();
    if (_pLearnOutputPipelineState)       _pLearnOutputPipelineState->release();
    if (_pLearnHiddenPipelineState)       _pLearnHiddenPipelineState->release();
    if (_pApplyUpdatesPipelineState)      _pApplyUpdatesPipelineState->release();
    
    // Release buffers.
    if (_pBuffer_x)               _pBuffer_x->release();
    if (_pBuffer_hidden)          _pBuffer_hidden->release();
    if (_pBuffer_y)               _pBuffer_y->release();
    if (_pBuffer_y_hat)           _pBuffer_y_hat->release();
    
    if (_pBuffer_W1)              _pBuffer_W1->release();
    if (_pBuffer_b1)              _pBuffer_b1->release();
    if (_pBuffer_prev_W1)         _pBuffer_prev_W1->release();
    if (_pBuffer_prev_b1)         _pBuffer_prev_b1->release();
    if (_pBuffer_W2)              _pBuffer_W2->release();
    if (_pBuffer_b2)              _pBuffer_b2->release();
    if (_pBuffer_prev_W2)         _pBuffer_prev_W2->release();
    if (_pBuffer_prev_b2)         _pBuffer_prev_b2->release();
    
    if (_pBuffer_M1)              _pBuffer_M1->release();
    if (_pBuffer_N1)              _pBuffer_N1->release();
    if (_pBuffer_M2)              _pBuffer_M2->release();
    if (_pBuffer_N2)              _pBuffer_N2->release();
    
    if (_pBuffer_randomness1)     _pBuffer_randomness1->release();
    if (_pBuffer_randomness2)     _pBuffer_randomness2->release();
    
    if (_pBuffer_error)           _pBuffer_error->release();
    if (_pBuffer_error_hidden)    _pBuffer_error_hidden->release();
    if (_pBuffer_prev_error_hidden)_pBuffer_prev_error_hidden->release();
    
    if (_pBuffer_WAccumulator1)   _pBuffer_WAccumulator1->release();
    if (_pBuffer_bAccumulator1)   _pBuffer_bAccumulator1->release();
    if (_pBuffer_WAccumulator2)   _pBuffer_WAccumulator2->release();
    if (_pBuffer_bAccumulator2)   _pBuffer_bAccumulator2->release();
    
    // Release functions.
    if (_pForwardLayerFn)         _pForwardLayerFn->release();
    if (_pForwardLayerFn2)        _pForwardLayerFn2->release();
    if (_pLearnOutputFn)          _pLearnOutputFn->release();
    if (_pLearnHiddenFn)          _pLearnHiddenFn->release();
    if (_pApplyUpdatesFn)         _pApplyUpdatesFn->release();
    
    // Release command queue & device.
    if (_pCommandQueue)           _pCommandQueue->release();
    if (_pDevice)                 _pDevice->release();
}

#pragma mark - Compute Pipeline Setup

void Computer::buildComputePipeline()
{
    printf("build compute pipeline (multi-layer)\n");
    _pCommandQueue = _pDevice->newCommandQueue();
    
    NS::Error* pError = nullptr;
    // Build library from kernel source.
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
    
    // Build function objects for each kernel.
    _pForwardLayerFn  = _pComputeLibrary->newFunction(NS::String::string("forward_layer", NS::UTF8StringEncoding));
    _pForwardLayerFn2 = _pComputeLibrary->newFunction(NS::String::string("forward_layer", NS::UTF8StringEncoding));
    _pLearnOutputFn   = _pComputeLibrary->newFunction(NS::String::string("learn_output_layer", NS::UTF8StringEncoding));
    _pLearnHiddenFn   = _pComputeLibrary->newFunction(NS::String::string("learn_hidden_layer", NS::UTF8StringEncoding));
    _pApplyUpdatesFn  = _pComputeLibrary->newFunction(NS::String::string("apply_updates", NS::UTF8StringEncoding));
    
    assert(_pForwardLayerFn);
    assert(_pForwardLayerFn2);
    assert(_pLearnOutputFn);
    assert(_pLearnHiddenFn);
    assert(_pApplyUpdatesFn);
    
    // Build pipeline states.
    _pForwardLayerPipelineState  = _pDevice->newComputePipelineState(_pForwardLayerFn,  &pError);
    _pForwardLayerPipelineState2 = _pDevice->newComputePipelineState(_pForwardLayerFn2, &pError);
    _pLearnOutputPipelineState   = _pDevice->newComputePipelineState(_pLearnOutputFn,   &pError);
    _pLearnHiddenPipelineState   = _pDevice->newComputePipelineState(_pLearnHiddenFn,   &pError);
    _pApplyUpdatesPipelineState  = _pDevice->newComputePipelineState(_pApplyUpdatesFn,  &pError);
    
    if (!_pForwardLayerPipelineState ||
        !_pForwardLayerPipelineState2 ||
        !_pLearnOutputPipelineState ||
        !_pLearnHiddenPipelineState ||
        !_pApplyUpdatesPipelineState) {
        std::cerr << "Error building multi-layer pipeline state." << pError->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    _pComputeLibrary->release();
}

#pragma mark - Buffer Setup

void Computer::buildBuffers()
{
    printf("buildBuffers (multi-layer)\n");
    
    // Define dimension values.
    uint m1 = input_dim;
    uint n1 = hidden_dim;
    uint m2 = hidden_dim;
    uint n2 = output_dim;
    
    // Buffer for input x.
    _pBuffer_x = _pDevice->newBuffer(x.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_x->contents(), x.get_data_buffer(), x.get_num_data() * sizeof(float));
    _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
    
    // Buffer for hidden activations.
    _pBuffer_hidden = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                          MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_hidden->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_hidden->didModifyRange(NS::Range::Make(0, _pBuffer_hidden->length()));
    
    // Buffer for output activations y.
    _pBuffer_y = _pDevice->newBuffer(output_dim * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_y->contents(), 0, output_dim * sizeof(float));
    _pBuffer_y->didModifyRange(NS::Range::Make(0, _pBuffer_y->length()));
    
    // Buffer for target output y_hat.
    _pBuffer_y_hat = _pDevice->newBuffer(y_hat.get_num_data() * sizeof(float),
                                         MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_y_hat->contents(), y_hat.get_data_buffer(), y_hat.get_num_data() * sizeof(float));
    _pBuffer_y_hat->didModifyRange(NS::Range::Make(0, _pBuffer_y_hat->length()));
    
    // Buffers for Layer 1 weights and biases.
    _pBuffer_W1 = _pDevice->newBuffer(W1.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W1->contents(), W1.get_data_buffer(), W1.get_num_data() * sizeof(float));
    _pBuffer_W1->didModifyRange(NS::Range::Make(0, _pBuffer_W1->length()));
    
    _pBuffer_b1 = _pDevice->newBuffer(b1.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b1->contents(), b1.get_data_buffer(), b1.get_num_data() * sizeof(float));
    _pBuffer_b1->didModifyRange(NS::Range::Make(0, _pBuffer_b1->length()));
    
    
    _pBuffer_prev_W1 = _pDevice->newBuffer(W1.get_num_data() * sizeof(float),
                                           MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_prev_W1->contents(), W1.get_data_buffer(), W1.get_num_data() * sizeof(float));
    _pBuffer_prev_W1->didModifyRange(NS::Range::Make(0, _pBuffer_prev_W1->length()));
    
    _pBuffer_prev_b1 = _pDevice->newBuffer(b1.get_num_data() * sizeof(float),
                                           MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_prev_b1->contents(), b1.get_data_buffer(), b1.get_num_data() * sizeof(float));
    _pBuffer_prev_b1->didModifyRange(NS::Range::Make(0, _pBuffer_prev_b1->length()));
    
    
    // Buffers for Layer 2 weights and biases.
    _pBuffer_W2 = _pDevice->newBuffer(W2.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W2->contents(), W2.get_data_buffer(), W2.get_num_data() * sizeof(float));
    _pBuffer_W2->didModifyRange(NS::Range::Make(0, _pBuffer_W2->length()));
    
    _pBuffer_b2 = _pDevice->newBuffer(b2.get_num_data() * sizeof(float),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b2->contents(), b2.get_data_buffer(), b2.get_num_data() * sizeof(float));
    _pBuffer_b2->didModifyRange(NS::Range::Make(0, _pBuffer_b2->length()));
    
    
    _pBuffer_prev_W2 = _pDevice->newBuffer(W2.get_num_data() * sizeof(float),
                                           MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_prev_W2->contents(), W2.get_data_buffer(), W2.get_num_data() * sizeof(float));
    _pBuffer_prev_W2->didModifyRange(NS::Range::Make(0, _pBuffer_prev_W2->length()));
    
    _pBuffer_prev_b2 = _pDevice->newBuffer(b2.get_num_data() * sizeof(float),
                                           MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_prev_b2->contents(), b2.get_data_buffer(), b2.get_num_data() * sizeof(float));
    _pBuffer_prev_b2->didModifyRange(NS::Range::Make(0, _pBuffer_prev_b2->length()));
    
    
    // Dimension buffers for Layer 1.
    _pBuffer_M1 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M1->contents(), &m1, sizeof(uint));
    _pBuffer_M1->didModifyRange(NS::Range::Make(0, _pBuffer_M1->length()));
    
    _pBuffer_N1 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N1->contents(), &n1, sizeof(uint));
    _pBuffer_N1->didModifyRange(NS::Range::Make(0, _pBuffer_N1->length()));
    
    // Dimension buffers for Layer 2.
    _pBuffer_M2 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M2->contents(), &m2, sizeof(uint));
    _pBuffer_M2->didModifyRange(NS::Range::Make(0, _pBuffer_M2->length()));
    
    _pBuffer_N2 = _pDevice->newBuffer(sizeof(uint),
                                      MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N2->contents(), &n2, sizeof(uint));
    _pBuffer_N2->didModifyRange(NS::Range::Make(0, _pBuffer_N2->length()));
    
    // Error buffers for backpropagation.
    _pBuffer_error = _pDevice->newBuffer(output_dim * sizeof(float),
                                         MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_error->contents(), 0, output_dim * sizeof(float));
    _pBuffer_error->didModifyRange(NS::Range::Make(0, _pBuffer_error->length()));
    
    _pBuffer_prev_error = _pDevice->newBuffer(output_dim * sizeof(float),
                                              MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_prev_error->contents(), 0, output_dim * sizeof(float));
    _pBuffer_prev_error->didModifyRange(NS::Range::Make(0, _pBuffer_prev_error->length()));
    
    _pBuffer_error_hidden = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                                MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_error_hidden->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_error_hidden->didModifyRange(NS::Range::Make(0, _pBuffer_error_hidden->length()));
    
    _pBuffer_prev_error_hidden = _pDevice->newBuffer(hidden_dim * sizeof(float),
                                                     MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_prev_error_hidden->contents(), 0, hidden_dim * sizeof(float));
    _pBuffer_prev_error_hidden->didModifyRange(NS::Range::Make(0, _pBuffer_prev_error_hidden->length()));
    
    // Accumulator buffers for gradient updates.
    // For Layer 1.
    _pBuffer_WAccumulator1 = _pDevice->newBuffer(W1.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_WAccumulator1->contents(), 0, W1.get_num_data() * sizeof(float));
    _pBuffer_WAccumulator1->didModifyRange(NS::Range::Make(0, _pBuffer_WAccumulator1->length()));
    
    _pBuffer_bAccumulator1 = _pDevice->newBuffer(b1.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_bAccumulator1->contents(), 0, b1.get_num_data() * sizeof(float));
    _pBuffer_bAccumulator1->didModifyRange(NS::Range::Make(0, _pBuffer_bAccumulator1->length()));
    
    // For Layer 2.
    _pBuffer_WAccumulator2 = _pDevice->newBuffer(W2.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_WAccumulator2->contents(), 0, W2.get_num_data() * sizeof(float));
    _pBuffer_WAccumulator2->didModifyRange(NS::Range::Make(0, _pBuffer_WAccumulator2->length()));
    
    _pBuffer_bAccumulator2 = _pDevice->newBuffer(b2.get_num_data() * sizeof(float),
                                                 MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_bAccumulator2->contents(), 0, b2.get_num_data() * sizeof(float));
    _pBuffer_bAccumulator2->didModifyRange(NS::Range::Make(0, _pBuffer_bAccumulator2->length()));
    
    
    _pBuffer_randomness1 = _pDevice->newBuffer(rand1.get_num_data() * sizeof(float),
                                               MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_randomness1->contents(), rand1.get_data_buffer(), sizeof(uint));
    _pBuffer_randomness1->didModifyRange(NS::Range::Make(0, _pBuffer_randomness1->length()));
    
    _pBuffer_randomness2 = _pDevice->newBuffer(rand2.get_num_data() * sizeof(float),
                                               MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_randomness2->contents(), rand2.get_data_buffer(), sizeof(uint));
    _pBuffer_randomness2->didModifyRange(NS::Range::Make(0, _pBuffer_randomness2->length()));
    
    areBuffersBuilt = true;
}

#pragma mark - Forward Pass

// The forward pass now consists of two kernel dispatches:
//   1. Compute hidden layer activations: forward_layer kernel (Layer 1)
//   2. Compute output activations: forward_layer kernel (Layer 2)
void Computer::computeForward(std::function<void()> onComplete)
{
    printf("computeForward (multi-layer)\n");
    
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Performing forward pass..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    // --- Layer 1: Input -> Hidden ---
    {
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(_pForwardLayerPipelineState);
        // Bind: input x, output hidden, W1, b1, dimensions M1 and N1.
        enc->setBuffer(_pBuffer_x,    0, 0);
        enc->setBuffer(_pBuffer_hidden, 0, 1);
        enc->setBuffer(_pBuffer_W1,   0, 2);
        enc->setBuffer(_pBuffer_b1,   0, 3);
        enc->setBuffer(_pBuffer_M1,   0, 4);
        enc->setBuffer(_pBuffer_N1,   0, 5);
        
        uint32_t threads = hidden_dim;
        MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
        MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
        enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        enc->endEncoding();
    }
    
    // --- Layer 2: Hidden -> Output ---
    {
        MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(_pForwardLayerPipelineState2);
        // Bind: input hidden, output y, W2, b2, dimensions M2 and N2.
        enc->setBuffer(_pBuffer_hidden, 0, 0);
        enc->setBuffer(_pBuffer_y,      0, 1);
        enc->setBuffer(_pBuffer_W2,     0, 2);
        enc->setBuffer(_pBuffer_b2,     0, 3);
        enc->setBuffer(_pBuffer_M2,     0, 4);
        enc->setBuffer(_pBuffer_N2,     0, 5);
        
        uint32_t threads = output_dim;
        MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
        MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
        enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        enc->endEncoding();
    }
    
    // Synchronize the output buffer for CPU access.
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pBuffer_y);
    blitEncoder->endEncoding();
    
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

#pragma mark - Learning (Backpropagation) & Apply Updates

// This method performs one iteration of backpropagation and then applies weight updates.
// 1. Output layer: learn_output_layer kernel computes output error and accumulates updates.
// 2. Hidden layer: learn_hidden_layer kernel backpropagates error and accumulates updates.
// 3. Then, apply_updates kernel is invoked for each layer.
void Computer::computeLearn(std::function<void()> onComplete)
{
    this->computeForward([this, onComplete](){
        printf("computeLearn (multi-layer)\n");
        
        if (!areBuffersBuilt) return;
        if (currentlyComputing) return;
        
        currentlyComputing = true;
        std::cout << "Performing learning (backpropagation)..." << std::endl;
        
        NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
        
        MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
        assert(cmdBuf);
        
        
            // --- Output Layer Backpropagation ---
            {
                MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(_pLearnOutputPipelineState);
                // Bind: input = hidden activations, W2, b2, output activation y,
                // target y_hat, error buffer for output, dimensions (M2,N2), accumulators.
                enc->setBuffer(_pBuffer_hidden,         0, 0);
                enc->setBuffer(_pBuffer_W2,             0, 1);
                enc->setBuffer(_pBuffer_b2,             0, 2);
                enc->setBuffer(_pBuffer_y,              0, 3);
                enc->setBuffer(_pBuffer_y_hat,          0, 4);
                enc->setBuffer(_pBuffer_error,          0, 5);
                enc->setBuffer(_pBuffer_prev_error,     0, 6);
                enc->setBuffer(_pBuffer_M2,             0, 7);
                enc->setBuffer(_pBuffer_N2,             0, 8);
                enc->setBuffer(_pBuffer_WAccumulator2,  0, 9);
                enc->setBuffer(_pBuffer_bAccumulator2,  0, 10);
                enc->setBuffer(_pBuffer_prev_W2,             0, 11);
                enc->setBuffer(_pBuffer_prev_b2,             0, 12);
                
                uint32_t threads = output_dim;
                MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
                MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
                enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
                enc->endEncoding();
            }
            
            // --- Hidden Layer Backpropagation ---
            {
                MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(_pLearnHiddenPipelineState);
                // Bind: input = x, W1, b1, output = hidden activations,
                // error output for hidden layer, next layer error (output layer),
                // next layer weights (W2), dimensions for hidden layer (M1,N1) and output layer (N2),
                // accumulators for Layer 1.
                enc->setBuffer(_pBuffer_x,               0, 0);
                enc->setBuffer(_pBuffer_W1,              0, 1);
                enc->setBuffer(_pBuffer_b1,              0, 2);
                enc->setBuffer(_pBuffer_hidden,          0, 3);
                enc->setBuffer(_pBuffer_error_hidden,    0, 4);
                enc->setBuffer(_pBuffer_prev_error_hidden,    0, 5);
                enc->setBuffer(_pBuffer_error,           0, 6);
                enc->setBuffer(_pBuffer_W2,              0, 7);
                enc->setBuffer(_pBuffer_M1,              0, 8);
                enc->setBuffer(_pBuffer_N1,              0, 9);
                enc->setBuffer(_pBuffer_N2,              0, 10);
                enc->setBuffer(_pBuffer_WAccumulator1,   0, 11);
                enc->setBuffer(_pBuffer_bAccumulator1,   0, 12);
                enc->setBuffer(_pBuffer_prev_W1,              0, 13);
                enc->setBuffer(_pBuffer_prev_b1,              0, 14);
                
                uint32_t threads = hidden_dim;
                MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
                MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
                enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
                enc->endEncoding();
            }
            
            
            // --- Apply Updates for Layer 2 ---
            {
                MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(_pApplyUpdatesPipelineState);
                // Bind: W2, b2, accumulators for Layer 2, dimensions M2 and N2.
                enc->setBuffer(_pBuffer_W2,             0, 0);
                enc->setBuffer(_pBuffer_b2,             0, 1);
                enc->setBuffer(_pBuffer_prev_W2,             0, 2);
                enc->setBuffer(_pBuffer_prev_b2,             0, 3);
                enc->setBuffer(_pBuffer_WAccumulator2,  0, 4);
                enc->setBuffer(_pBuffer_bAccumulator2,  0, 5);
                enc->setBuffer(_pBuffer_M2,             0, 6);
                enc->setBuffer(_pBuffer_N2,             0, 7);
                enc->setBuffer(_pBuffer_randomness1,    0, 8);
                
                uint32_t threads = output_dim;
                MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
                MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
                enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
                enc->endEncoding();
            }
            
            // --- Apply Updates for Layer 1 ---
            {
                MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(_pApplyUpdatesPipelineState);
                // Bind: W1, b1, accumulators for Layer 1, dimensions M1 and N1.
                enc->setBuffer(_pBuffer_W1,            0, 0);
                enc->setBuffer(_pBuffer_b1,            0, 1);
                enc->setBuffer(_pBuffer_prev_W1,            0, 2);
                enc->setBuffer(_pBuffer_prev_b1,            0, 3);
                enc->setBuffer(_pBuffer_WAccumulator1, 0, 4);
                enc->setBuffer(_pBuffer_bAccumulator1, 0, 5);
                enc->setBuffer(_pBuffer_M1,            0, 6);
                enc->setBuffer(_pBuffer_N1,            0, 7);
                enc->setBuffer(_pBuffer_randomness2,   0, 8);
                
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
    printf("computeLearnAndApplyUpdates (multi-layer) - iterations remaining = %d\n", (int)iterations);
    this->computeLearn([this, iterations]() {
        // Update input data for the next iteration.
        this->x.build([iterations](double x){ return inputFunction(x - iterations); });
        this->y_hat.build([iterations](double x){ return inputFunction(x - iterations); });
        
        std::memcpy(_pBuffer_x->contents(), x.get_data_buffer(), x.get_num_data() * sizeof(float));
        _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
        
        std::memcpy(_pBuffer_y_hat->contents(), y_hat.get_data_buffer(), y_hat.get_num_data() * sizeof(float));
        _pBuffer_y_hat->didModifyRange(NS::Range::Make(0, _pBuffer_y_hat->length()));
        
        if (iterations > 0) {
            this->computeLearnAndApplyUpdates(iterations - 1);
        }
    });
}

void Computer::computeForwardIterations(uint32_t iterations)
{
    printf("computeForwardIterations (multi-layer)\n");
    this->computeForward([this, iterations]() {
        printf("Forward Iterations remaining=%d\n", iterations);
        this->x.build([iterations](double x){ return inputFunction(x - iterations); });
        std::memcpy(_pBuffer_x->contents(), x.get_data_buffer(), x.get_num_data() * sizeof(float));

        // Synchronize GPU buffers to CPU.
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

#pragma mark - Logging & Output

void Computer::logError()
{
    float* error = static_cast<float*>(_pBuffer_error->contents());
    
    float avg_error = 0.0f;
    for (int i = 0; i < _pBuffer_error->length(); i++) {
        avg_error += error[i];
    }
    avg_error /= _pBuffer_error->length();
    printf("AVG OUTPUT ERROR: %f\n", avg_error);
    
    float* error_hidden = static_cast<float*>(_pBuffer_error_hidden->contents());
    
    float avg_error_hidden = 0.0f;
    for (int i = 0; i < _pBuffer_error_hidden->length(); i++) {
        avg_error_hidden += error_hidden[i];
    }
    avg_error_hidden /= _pBuffer_error_hidden->length();
    printf("AVG HIDDEN ERROR: %f\n", avg_error_hidden);
}

void Computer::logInformation(const std::string& filename, int remainingIterations)
{
    printf("logInformation\n");
    std::ofstream logFile(filename, std::ios::app); // Open file in append mode.
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file!" << std::endl;
        return;
    }
    
    logFile << "clf; hold on;" << std::endl;
    logFile << "ylim([-1 1]);" << std::endl;
    
    // For logging, we print the input and output values.
    float* inputPtr = static_cast<float*>(_pBuffer_x->contents());
    float* hiddenPtr = static_cast<float*>(_pBuffer_hidden->contents());
    float* outputPtr = static_cast<float*>(_pBuffer_y->contents());
    float* targetPtr = static_cast<float*>(_pBuffer_y_hat->contents());
    
    float* error = static_cast<float*>(_pBuffer_error->contents());
    
    float avg_error = 0.0f;
    for (int i = 0; i < _pBuffer_error->length(); i++) {
        avg_error += error[i];
    }
    avg_error /= _pBuffer_error->length();
    printf("AVG ERROR: %f\n", avg_error);
    
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
    // Log the results.
    logInformation(outputFileName, remainingIterations);
}

void Computer::clearOutput()
{
    printf("clearOutput\n");
    std::ofstream logFile(outputFileName, std::ios::trunc); // Open file in truncate mode.
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file!" << std::endl;
        return;
    }
    logFile << std::endl;
}

#pragma mark - User Input Handling

void Computer::keyPress(KeyPress* kp)
{
    if (kp->pressed) {
        keyState[kp->code] = true;
    } else {
        keyState.erase(kp->code);
    }
}

void Computer::handleKeyStateChange()
{
    printf("handleKeyStateChange\n");
    // 'F' triggers forward pass.
    {
        auto it = keyState.find(9); // Key code for 'F'
        if (it != keyState.end() && it->second) {
            this->clearOutput();
            this->computeForwardIterations(NUM_ITERATIONS);
        }
    }
    
    // 'L' triggers learn.
    {
        auto it = keyState.find(15); // Key code for 'L'
        if (it != keyState.end() && it->second) {
            this->computeLearnAndApplyUpdates(NUM_ITERATIONS);
        }
    }
    
    // 'C' clears output.
    {
        auto it = keyState.find(6); // Key code for 'C'
        if (it != keyState.end() && it->second) {
            this->clearOutput();
        }
    }
}
