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

const int DATA_SOURCE_MAX_VECTORS_PER_ROW = 256;

Computer::Computer(MTL::Device* pDevice) :
dataSource(DATA_SOURCE_MAX_VECTORS_PER_ROW),
_pDevice(pDevice->retain()),
_pCompileOptions(),
areBuffersBuilt(false),
currentlyComputing(false)
{
    _pCommandQueue = _pDevice->newCommandQueue();
    buildComputePipeline();
    
    // Asynchronous data build; once done, buffers are built on the main queue.
    dataSource.buildAsync([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            this->buildBuffers();
            areBuffersBuilt = true;
        });
    });
    
    _semaphore = dispatch_semaphore_create(Computer::kMaxFramesInFlight);
}

Computer::~Computer()
{
    if (_pComputePipelineState) _pComputePipelineState->release();
    if (_pArgBuffer) _pArgBuffer->release();
    if (_pBufferA) _pBufferA->release();
    if (_pBufferB) _pBufferB->release();
    if (_pResultBuffer) _pResultBuffer->release();
    
    _pCommandQueue->release();
    _pDevice->release();
    _pComputeFn->release();
}

void Computer::buildComputePipeline()
{
    NS::Error* pError = nullptr;
    _pComputeLibrary = _pDevice->newLibrary(NS::String::string(kernels::addArrayKernelSrc, NS::UTF8StringEncoding), _pCompileOptions, &pError);
    
    if (!_pComputeLibrary)
    {
        std::cerr << "Compute library error: " << pError->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    
    _pComputeFn = _pComputeLibrary->newFunction(NS::String::string("add_arrays", NS::UTF8StringEncoding));
    
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
    // Assuming dataSource provides two float arrays for inA and inB.
    const size_t numElements = dataSource.get_num_data();
    const size_t dataSize = numElements * sizeof(simd::float3);
    
    // Create buffer for inA.
    _pBufferA = _pDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    std::memcpy(_pBufferA->contents(), dataSource.get_data_buffer(), dataSize);
    _pBufferA->didModifyRange(NS::Range::Make(0, dataSize));
    
    // Create buffer for inB.
    _pBufferB = _pDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    std::memcpy(_pBufferB->contents(), dataSource.get_data_buffer(), dataSize);
    _pBufferB->didModifyRange(NS::Range::Make(0, dataSize));
    
    // Create output buffer for result.
    _pResultBuffer = _pDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    _pResultBuffer->didModifyRange(NS::Range::Make(0, dataSize));
    
    std::memcpy(_pResultBuffer->contents(), dataSource.get_data_buffer(), dataSize);
    
    

    // Create an argument encoder for the Buffers struct (defined in the shader) at buffer index 0.
    MTL::ArgumentEncoder* pArgEncoder = _pComputeFn->newArgumentEncoder(0);
    _pArgBuffer = _pDevice->newBuffer(pArgEncoder->encodedLength(), MTL::ResourceStorageModeManaged);
    pArgEncoder->setArgumentBuffer(_pArgBuffer, 0);
    
    // Bind the individual buffers into the argument buffer at their designated [[id]] slots.
    pArgEncoder->setBuffer(_pBufferA, 0, 0);      // inA at [[id(0)]]
    pArgEncoder->setBuffer(_pBufferB, 0, 1);      // inB at [[id(1)]]
    pArgEncoder->setBuffer(_pResultBuffer, 0, 2); // result at [[id(2)]]
    
    _pArgBuffer->didModifyRange(NS::Range::Make(0, _pArgBuffer->length()));
    
    pArgEncoder->release();
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
    
    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
        _pResultBuffer->didModifyRange(NS::Range(0, _pResultBuffer->length()));
        
        this->extractResults();
        
        std::cout << "Done Computing." << std::endl;
        currentlyComputing = false;
    });
    
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    
    enc->setComputePipelineState(_pComputePipelineState);
    enc->setBuffer(_pArgBuffer, 0, 0);  // Bind the argument buffer at index 0
    
    MTL::Size threadsPerThreadgroup = MTL::Size{1024, 1, 1};
    MTL::Size threadgroups = MTL::Size{
    (unsigned int)ceil(((DATA_SOURCE_MAX_VECTORS_PER_ROW*DATA_SOURCE_MAX_VECTORS_PER_ROW) / 1024)),
    1,
    1
    };
    
    extractResults();
    
    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    
    enc->endEncoding();
    
    MTL::BlitCommandEncoder* blitEncoder = cmdBuf->blitCommandEncoder();
    blitEncoder->synchronizeResource(_pResultBuffer);
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

void Computer::extractResults() {    
    //simd::float3* a = static_cast<simd::float3*>(_pBufferA->contents());
    //simd::float3* b = static_cast<simd::float3*>(_pBufferB->contents());
    simd::float3* result = static_cast<simd::float3*>(_pResultBuffer->contents());
    
    uint64_t length = _pResultBuffer->length() / sizeof(float);
    
    simd::float3 sum = simd::float3{0, 0, 0};
    for (unsigned long index = 0; index < length; index++)
    {
        sum += *result;
    }
    printf("<%f,%f,%f>", result->x, result->y, result->z);
    
    // Unmap the buffer when done
    _pResultBuffer->didModifyRange(NS::Range(0, _pResultBuffer->length()));
}

#pragma endregion Computer
