/*
 * computer.cpp
 * Created by James Couch on 2024-12-07.
 */

#include <simd/simd.h>
#include <cmath>
#include <iostream>

#include "computer.h"
#include "data-source.h"
#include "kernels.h"

// stb_image for loading PNG/JPG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma mark - Computer
#pragma region Computer {

const int DATA_SOURCE_WIDTH = 300;
const int DATA_SOURCE_MAX_VECTORS_PER_ROW = 300;

Computer::Computer(MTL::Device* pDevice) :
    dataSource(DATA_SOURCE_MAX_VECTORS_PER_ROW, DATA_SOURCE_WIDTH),
    _pDevice(pDevice->retain()),
    _pCompileOptions(),
    areBuffersBuilt(false)
{
    _pCommandQueue = _pDevice->newCommandQueue();
    buildComputePipeline();

    // Asynchronous data build
    dataSource.buildAsync([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            buildBuffers();
            areBuffersBuilt = true;
        });
    });

    _semaphore = dispatch_semaphore_create(Computer::kMaxFramesInFlight);
}

Computer::~Computer()
{
    if (_pComputePipelineState) _pComputePipelineState->release();
    if (_pArgBuffer) _pArgBuffer->release();
    if (_pInputBuffer) _pInputBuffer->release();
    
    _pCommandQueue->release();
    _pDevice->release();
    _pComputeLibrary->release();
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
}

void Computer::buildBuffers()
{
    const size_t numData = dataSource.get_num_data();
    const size_t dataSize = numData * sizeof(simd::float3);

    // Allocate buffer
    _pInputBuffer = _pDevice->newBuffer(dataSize, MTL::ResourceStorageModeManaged);
    std::memcpy(_pInputBuffer->contents(), dataSource.get_data_buffer(), dataSize);
    _pInputBuffer->didModifyRange(NS::Range::Make(0, dataSize));
    
    MTL::ArgumentEncoder* pArgEncoder = _pComputeFn->newArgumentEncoder(0);

    _pArgBuffer = _pDevice->newBuffer(pArgEncoder->encodedLength(), MTL::ResourceStorageModeManaged);
    pArgEncoder->setArgumentBuffer(_pArgBuffer, 0);
    pArgEncoder->setBuffer(_pInputBuffer, 0, 0);
    _pArgBuffer->didModifyRange(NS::Range::Make(0, _pArgBuffer->length()));

    pArgEncoder->release();
}

void Computer::compute()
{
    if (!areBuffersBuilt) return;

    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();

    Computer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
    });

    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();
    doCompute(enc);

    enc->endEncoding();
    cmdBuf->commit();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    pPool->release();
}

void Computer::doCompute(MTL::ComputeCommandEncoder* enc)
{
    enc->setComputePipelineState(_pComputePipelineState);
    enc->setBuffer(_pArgBuffer, 0, 0);
    enc->setBuffer(_pInputBuffer, 0, 1);
    
    MTL::Size threadsPerThreadgroup = MTL::Size{16, 16, 1};
    MTL::Size threadgroups = MTL::Size{
        (DATA_SOURCE_WIDTH + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
        (DATA_SOURCE_WIDTH + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
        1
    };

    enc->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
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


#pragma endregion Computer
