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
#pragma region Computer {

const int N = 256;

Computer::Computer(MTL::Device* pDevice) :
x(N, 1),
W(N, N),
b(N, 1),
_pDevice(pDevice->retain()),
_pCompileOptions(),
areBuffersBuilt(false),
currentlyComputing(false)
{
    buildComputePipeline();
    
    // Callback hell
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
    if (_pComputePipelineState) _pComputePipelineState->release();
    
    if (_pBuffer_x)     _pBuffer_x->release();
    if (_pBuffer_W)     _pBuffer_W->release();
    if (_pBuffer_y)     _pBuffer_y->release();
    if (_pBuffer_M)     _pBuffer_M->release();
    if (_pBuffer_N)     _pBuffer_N->release();
    
    _pCommandQueue->release();
    _pDevice->release();
    _pComputeFn->release();
}

void Computer::buildComputePipeline()
{
    _pCommandQueue = _pDevice->newCommandQueue();
    
    NS::Error* pError = nullptr;
    _pComputeLibrary = _pDevice->newLibrary(NS::String::string(kernels::addArrayKernelSrc, NS::UTF8StringEncoding), _pCompileOptions, &pError);
    
    if (!_pComputeLibrary)
    {
        std::cerr << "Compute library error: " << pError->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    _pComputeFn = _pComputeLibrary->newFunction(NS::String::string("forward", NS::UTF8StringEncoding));
    
    NS::Error* pErr2 = nullptr;
    _pComputePipelineState = _pDevice->newComputePipelineState(_pComputeFn, &pErr2);
    if (!_pComputePipelineState)
    {
        std::cerr << "Compute pipeline error: " << pErr2->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    _pComputeLibrary->release();
    //_pComputeFn->release();
}

void Computer::buildBuffers()
{
    int m = N;
    int n = N;
    
    // Assuming dataSource provides two float arrays for inA and inB.
    _pBuffer_x = _pDevice->newBuffer(x.get_num_data() * sizeof(float), MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_x->contents(), x.get_data_buffer(), x.get_num_data() * sizeof(float));
    _pBuffer_x->didModifyRange(NS::Range::Make(0, _pBuffer_x->length()));
    
    _pBuffer_W = _pDevice->newBuffer(W.get_num_data() * sizeof(float), MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_W->contents(), W.get_data_buffer(), W.get_num_data() * sizeof(float));
    _pBuffer_W->didModifyRange(NS::Range::Make(0, _pBuffer_W->length()));
    
    _pBuffer_b = _pDevice->newBuffer(b.get_num_data() * sizeof(float), MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_b->contents(), b.get_data_buffer(), b.get_num_data() * sizeof(float));
    _pBuffer_b->didModifyRange(NS::Range::Make(0, _pBuffer_b->length()));
    
    _pBuffer_M = _pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_M->contents(), &m, sizeof(int));
    _pBuffer_M->didModifyRange(NS::Range::Make(0, _pBuffer_M->length()));
    
    _pBuffer_N = _pDevice->newBuffer(sizeof(int), MTL::ResourceStorageModeManaged);
    std::memcpy(_pBuffer_N->contents(), &n, sizeof(int));
    _pBuffer_N->didModifyRange(NS::Range::Make(0, _pBuffer_N->length()));
    

    // Create output buffer for result.
    _pBuffer_y = _pDevice->newBuffer(x.get_num_data() * sizeof(float), MTL::ResourceStorageModeManaged); //FIXME - assumes square
    _pBuffer_y->didModifyRange(NS::Range::Make(0, _pBuffer_y->length()));
        
    areBuffersBuilt = true;
}

void Computer::compute()
{
    if (!areBuffersBuilt) return;
    if (currentlyComputing) return;
    
    currentlyComputing = true;
    std::cout << "Computing..." << std::endl;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    assert(cmdBuf);
    
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(_pComputePipelineState);
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
        
        std::cout << "Done Computing." << std::endl;
        currentlyComputing = false;
    });
    
    const uint32_t maxThreads = 1024;
    MTL::Size threadsPerThreadgroup = MTL::Size{maxThreads, 1, 1};
    MTL::Size threadgroups = MTL::Size{
        (unsigned int)ceil(((N*N) / maxThreads)),
        1,
        1
    };
    
    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    enc->endEncoding();
    
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pBuffer_y);
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

void Computer::handleKeyStateChange() {
    auto it = keyState.find(6); // Key: C
    if (it != keyState.end()) {
        if (it->second) {
            this->compute();
        }
    }
}

void Computer::extractResults(MTL::Buffer* pBuffer) {
    simd::float3* result = static_cast<simd::float3*>(pBuffer->contents());
    
    uint64_t length = pBuffer->length() / sizeof(simd::float3);
    
    simd::float3 sum = simd::float3{0, 0, 0};
    for (unsigned long index = 0; index < length; index++)
    {
        sum += result[index];
    }
    printf("<%f,%f,%f>\n", sum.x, sum.y, sum.z);
    
    // Unmap the buffer when done
    pBuffer->didModifyRange(NS::Range(0, pBuffer->length()));
}

#pragma endregion Computer
