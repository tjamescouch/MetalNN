//
//  optimizable-layr.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include "optimizable-layer.h"

OptimizableLayer::OptimizableLayer()
    : bufferM_(nullptr), bufferV_(nullptr), adamTimestep_(0),
      adamPipelineState_(nullptr),
      learning_rate_(0.001f), beta1_(0.9f), beta2_(0.999f), epsilon_(1e-8f)
{ }

OptimizableLayer::~OptimizableLayer() {
    if (bufferM_) bufferM_->release();
    if (bufferV_) bufferV_->release();
    if (adamPipelineState_) adamPipelineState_->release();
}

void OptimizableLayer::buildAdamPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    MTL::Function* adamFunction = library->newFunction(NS::String::string("adam_kernel", NS::ASCIIStringEncoding));
    adamPipelineState_ = device->newComputePipelineState(adamFunction, &error);
    if (!adamPipelineState_) {
        throw std::runtime_error(error->localizedDescription()->utf8String());
    }
}

void OptimizableLayer::buildAdamBuffers(MTL::Device* device, size_t paramSize) {
    bufferM_ = device->newBuffer(paramSize, MTL::ResourceStorageModeShared);
    bufferV_ = device->newBuffer(paramSize, MTL::ResourceStorageModeShared);
    adamTimestep_ = 0;
    memset(bufferM_->contents(), 0, paramSize);
    memset(bufferV_->contents(), 0, paramSize);
}

void OptimizableLayer::encodeAdamKernel(MTL::ComputeCommandEncoder* encoder,
                                        MTL::Buffer* params,
                                        MTL::Buffer* grads,
                                        uint32_t paramCount) {
    adamTimestep_++;

    encoder->setComputePipelineState(adamPipelineState_);
    encoder->setBuffer(params, 0, 0);
    encoder->setBuffer(grads, 0, 1);
    encoder->setBuffer(bufferM_, 0, 2);
    encoder->setBuffer(bufferV_, 0, 3);
    encoder->setBytes(&adamTimestep_, sizeof(uint32_t), 4);
    encoder->setBytes(&learning_rate_, sizeof(float), 5);
    encoder->setBytes(&beta1_, sizeof(float), 6);
    encoder->setBytes(&beta2_, sizeof(float), 7);
    encoder->setBytes(&epsilon_, sizeof(float), 8);

    MTL::Size gridSize = MTL::Size(paramCount, 1, 1);
    NS::UInteger threadGroupSize = adamPipelineState_->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    encoder->dispatchThreads(gridSize, threadgroupSize);
}
