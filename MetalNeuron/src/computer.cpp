/*
 * computer.cpp
 * Created by James Couch on 2024-12-07.
 */

#include <simd/simd.h>
#include <cmath>
#include <iostream>
#include <cassert>

#include "computer.h"
#include "data-source.h"
#include "kernels.h"

// stb_image for loading PNG/JPG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma mark - Computer
#pragma region Computer

// Example dimension
const int N = 256;

Computer::Computer(MTL::Device* pDevice)
: x(N, 1),
W(N, N),
b(N, 1),
_pDevice(pDevice->retain()),
_pCompileOptions(),
areBuffersBuilt(false),
currentlyComputing(false)
{
    buildComputePipeline();
    
    // Initialize data sources asynchronously
    x.buildAsync([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            W.initRandomAsync([this]() {
                dispatch_async(dispatch_get_main_queue(), ^{
                    b.initRandomAsync([this]() {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            this->buildBuffers();
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
    // Release pipeline states
    if (_pForwardComputePipelineState)      _pForwardComputePipelineState->release();
    if (_pLearnComputePipelineState)        _pLearnComputePipelineState->release();
    if (_pApplyUpdatesComputePipelineState) _pApplyUpdatesComputePipelineState->release();
    
    // Release buffers
    if (_pBuffer_x)            _pBuffer_x->release();
    if (_pBuffer_W)            _pBuffer_W->release();
    if (_pBuffer_b)            _pBuffer_b->release();
    if (_pBuffer_y)            _pBuffer_y->release();
    if (_pBuffer_M)            _pBuffer_M->release();
    if (_pBuffer_N)            _pBuffer_N->release();
    if (_pBuffer_error)        _pBuffer_error->release();
    if (_pBuffer_WAccumulator) _pBuffer_WAccumulator->release();
    if (_pBuffer_bAccumulator) _pBuffer_bAccumulator->release();
    
    // Release functions
    if (_pForwardFn)      _pForwardFn->release();
    if (_pLearnFn)        _pLearnFn->release();
    if (_pApplyUpdatesFn) _pApplyUpdatesFn->release();
    
    // Release command queue & device
    if (_pCommandQueue)   _pCommandQueue->release();
    if (_pDevice)         _pDevice->release();
}

void Computer::buildComputePipeline()
{
    _pCommandQueue = _pDevice->newCommandQueue();
    
    NS::Error* pError = nullptr;
    // Build library from kernel source
    _pComputeLibrary = _pDevice->newLibrary(
                                            NS::String::string(kernels::nnKernelSrc, NS::UTF8StringEncoding),
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
    
    // Build functions
    _pForwardFn      = _pComputeLibrary->newFunction(NS::String::string("forward",       NS::UTF8StringEncoding));
    _pLearnFn        = _pComputeLibrary->newFunction(NS::String::string("learn",         NS::UTF8StringEncoding));
    _pApplyUpdatesFn = _pComputeLibrary->newFunction(NS::String::string("apply_updates", NS::UTF8StringEncoding));
    
    assert(_pForwardFn);
    assert(_pLearnFn);
    assert(_pApplyUpdatesFn);
    
    // Build pipeline states
    _pForwardComputePipelineState = _pDevice->newComputePipelineState(_pForwardFn, &pError);
    if (!_pForwardComputePipelineState)
    {
        std::cerr << "Compute pipeline error (forward): "
        << pError->localizedDescription()->utf8String()
        << std::endl;
        assert(false);
    }
    
    _pLearnComputePipelineState = _pDevice->newComputePipelineState(_pLearnFn, &pError);
    if (!_pLearnComputePipelineState)
    {
        std::cerr << "Compute pipeline error (learn): "
        << pError->localizedDescription()->utf8String()
        << std::endl;
        assert(false);
    }
    
    _pApplyUpdatesComputePipelineState = _pDevice->newComputePipelineState(_pApplyUpdatesFn, &pError);
    if (!_pApplyUpdatesComputePipelineState)
    {
        std::cerr << "Compute pipeline error (apply_updates): "
        << pError->localizedDescription()->utf8String()
        << std::endl;
        assert(false);
    }
    
    _pComputeLibrary->release();
}

void Computer::buildBuffers()
{
    uint m = N;
    uint n = N;
    
    // Buffer for x
    _pBuffer_x = _pDevice->newBuffer(x.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_x->contents(), x.get_data_buffer(),
                x.get_num_data() * sizeof(float));
    _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
    
    // Buffer for W
    _pBuffer_W = _pDevice->newBuffer(W.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W->contents(), W.get_data_buffer(),
                W.get_num_data() * sizeof(float));
    _pBuffer_W->didModifyRange(NS::Range::Make(0, _pBuffer_W->length()));
    
    // Buffer for b
    _pBuffer_b = _pDevice->newBuffer(b.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b->contents(), b.get_data_buffer(),
                b.get_num_data() * sizeof(float));
    _pBuffer_b->didModifyRange(NS::Range::Make(0, _pBuffer_b->length()));
    
    // Buffer for M
    _pBuffer_M = _pDevice->newBuffer(sizeof(int),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M->contents(), &m, sizeof(int));
    _pBuffer_M->didModifyRange(NS::Range::Make(0, _pBuffer_M->length()));
    
    // Buffer for N
    _pBuffer_N = _pDevice->newBuffer(sizeof(int),
                                     MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N->contents(), &n, sizeof(int));
    _pBuffer_N->didModifyRange(NS::Range::Make(0, _pBuffer_N->length()));
    
    // Output buffer y
    // Size matches x or b (a single row of outputs)
    _pBuffer_y = _pDevice->newBuffer(x.get_num_data() * sizeof(float),
                                     MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_y->contents(), 0, x.get_num_data() * sizeof(float));
    _pBuffer_y->didModifyRange(NS::Range::Make(0, _pBuffer_y->length()));
    
    // Error buffer
    _pBuffer_error = _pDevice->newBuffer(n * sizeof(float),
                                         MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_error->contents(), 0, n * sizeof(float));
    _pBuffer_error->didModifyRange(NS::Range::Make(0, _pBuffer_error->length()));
    
    // Weight accumulator buffer
    _pBuffer_WAccumulator = _pDevice->newBuffer(W.get_num_data() * sizeof(float),
                                                MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_WAccumulator->contents(), 0, W.get_num_data() * sizeof(float));
    _pBuffer_WAccumulator->didModifyRange(NS::Range::Make(0, _pBuffer_WAccumulator->length()));
    
    // Bias accumulator buffer
    _pBuffer_bAccumulator = _pDevice->newBuffer(b.get_num_data() * sizeof(float),
                                                MTL::ResourceStorageModeManaged);
    std::memset(_pBuffer_bAccumulator->contents(), 0, b.get_num_data() * sizeof(float));
    _pBuffer_bAccumulator->didModifyRange(NS::Range::Make(0, _pBuffer_bAccumulator->length()));
    
    areBuffersBuilt = true;
}

void Computer::computeForward()
{
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Forward..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(_pForwardComputePipelineState);
    
    // Bind arguments
    enc->setBuffer(_pBuffer_x, 0, 0);
    enc->setBuffer(_pBuffer_W, 0, 1);
    enc->setBuffer(_pBuffer_b, 0, 2);
    enc->setBuffer(_pBuffer_y, 0, 3);
    enc->setBuffer(_pBuffer_M, 0, 4);
    enc->setBuffer(_pBuffer_N, 0, 5);
    
    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
        this->extractResults(_pBuffer_y);
        std::cout << "Done Forward." << std::endl;
        currentlyComputing = false;
    });
    
    const uint32_t maxThreads = 1024;
    MTL::Size threadsPerThreadgroup = MTL::Size{maxThreads, 1, 1};
    MTL::Size threadgroups = MTL::Size{
        (unsigned int)ceil(((N * N) / double(maxThreads))),
        1,
        1
    };
    
    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    enc->endEncoding();
    
    // Sync the result
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pBuffer_y);
    blitEncoder->endEncoding();
    
    cmdBuf->commit();
    
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    pPool->release();
}

void Computer::computeLearn()
{
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Learning..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    // Example: fill _pBuffer_error with CPU-based error calculation
    {
        float* errPtr = reinterpret_cast<float*>(_pBuffer_error->contents());
        // For demonstration, we'll just create some dummy sinusoidal error
        for(int i = 0; i < N; i++) {
            errPtr[i] = std::sin(i * 0.01f);
        }
        _pBuffer_error->didModifyRange(NS::Range::Make(0, _pBuffer_error->length()));
    }
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(_pLearnComputePipelineState);
    
    // Bind arguments
    enc->setBuffer(_pBuffer_x,            0, 0);
    enc->setBuffer(_pBuffer_W,            0, 1);
    enc->setBuffer(_pBuffer_b,            0, 2);
    enc->setBuffer(_pBuffer_y,            0, 3);
    enc->setBuffer(_pBuffer_error,        0, 4);
    enc->setBuffer(_pBuffer_M,            0, 5);
    enc->setBuffer(_pBuffer_N,            0, 6);
    enc->setBuffer(_pBuffer_WAccumulator, 0, 7);
    enc->setBuffer(_pBuffer_bAccumulator, 0, 8);
    
    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
        // Optionally read any intermediate results from y
        this->extractResults(_pBuffer_y);
        std::cout << "Done Learning." << std::endl;
        currentlyComputing = false;
    });
    
    const uint32_t maxThreads = 1024;
    MTL::Size threadsPerThreadgroup = MTL::Size{maxThreads, 1, 1};
    MTL::Size threadgroups = MTL::Size{
        (unsigned int)ceil(((N * N) / double(maxThreads))),
        1,
        1
    };
    
    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    enc->endEncoding();
    
    // Sync the result for debugging
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pBuffer_y);
    blitEncoder->endEncoding();
    
    cmdBuf->commit();
    
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    pPool->release();
}

void Computer::computeApplyUpdates()
{
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Applying updates..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(_pApplyUpdatesComputePipelineState);
    
    // Bind arguments
    enc->setBuffer(_pBuffer_W,            0, 0);
    enc->setBuffer(_pBuffer_b,            0, 1);
    enc->setBuffer(_pBuffer_WAccumulator, 0, 2);
    enc->setBuffer(_pBuffer_bAccumulator, 0, 3);
    enc->setBuffer(_pBuffer_M,            0, 4);
    enc->setBuffer(_pBuffer_N,            0, 5);
    
    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
        std::cout << "Done Applying Updates." << std::endl;
        currentlyComputing = false;
    });
    
    const uint32_t maxThreads = 1024;
    MTL::Size threadsPerThreadgroup = MTL::Size{maxThreads, 1, 1};
    MTL::Size threadgroups = MTL::Size{
        (unsigned int)ceil(((N * N) / double(maxThreads))),
        1,
        1
    };
    
    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    enc->endEncoding();
    
    // Sync W and b to read them on CPU if desired
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pBuffer_W);
    blitEncoder->synchronizeResource(_pBuffer_b);
    blitEncoder->endEncoding();
    
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    pPool->release();
}

void Computer::keyPress(KeyPress* kp)
{
    if (kp->pressed)
    {
        keyState[kp->code] = true;
    }
    else
    {
        keyState.erase(kp->code);
    }
}

void Computer::handleKeyStateChange()
{
    // For example, 'F' triggers forward
    {
        auto it = keyState.find(9); // Key code for 'F'
        if (it != keyState.end()) {
            if (it->second) {
                this->computeForward();
            }
        }
    }
    
    // 'L' triggers learn
    {
        auto it = keyState.find(15); // Key code for 'L'
        if (it != keyState.end()) {
            if (it->second) {
                this->computeLearn();
            }
        }
    }
    
    // Let's say 'U' triggers apply updates (pick the correct key code as needed)
    {
        auto it = keyState.find(32); // Or whatever code you want for 'U'
        if (it != keyState.end()) {
            if (it->second) {
                this->computeApplyUpdates();
            }
        }
    }
}

void Computer::extractResults(MTL::Buffer* pBuffer)
{
    // Example: interpret the result buffer as an array of floats
    float* result = static_cast<float*>(pBuffer->contents());
    uint64_t length = pBuffer->length() / sizeof(float);
    
    //double sum = 0.0;
    for (unsigned long index = 0; index < length; index++)
    {
        //sum += result[index];
        printf("r[%lu] = %f\n", index, result[index]);
    }
    //printf("sum of result = %f\n", sum);
    
    // Let Metal know we modified the buffer (required in MTL::ResourceStorageModeManaged)
    pBuffer->didModifyRange(NS::Range(0, pBuffer->length()));
}

#pragma endregion Computer
