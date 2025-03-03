//
//  adam-optimizer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-03.
//
#include "adam-optimizer.h"

AdamOptimizer::AdamOptimizer(float lr, float beta1, float beta2, float epsilon)
    : bufferGradients_(nullptr), bufferM_(nullptr), bufferV_(nullptr),
      pipelineState_(nullptr), timestep_(0),
      learningRate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

void AdamOptimizer::buildBuffers(MTL::Device* device, size_t paramSize) {
    bufferGradients_ = device->newBuffer(paramSize, MTL::ResourceStorageModeShared);
    bufferM_ = device->newBuffer(paramSize, MTL::ResourceStorageModeShared);
    bufferV_ = device->newBuffer(paramSize, MTL::ResourceStorageModeShared);

    memset(bufferGradients_->contents(), 0, paramSize);
    memset(bufferM_->contents(), 0, paramSize);
    memset(bufferV_->contents(), 0, paramSize);
}

void AdamOptimizer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    auto adamFunc = library->newFunction(NS::String::string("adam_kernel", NS::UTF8StringEncoding));
    pipelineState_ = device->newComputePipelineState(adamFunc, &error);
    if (!pipelineState_) {
        throw std::runtime_error(error->localizedDescription()->utf8String());
    }
}

void AdamOptimizer::encode(MTL::ComputeCommandEncoder* encoder,
                           MTL::Buffer* params,
                           uint32_t paramCount) {
    timestep_++;

    encoder->setComputePipelineState(pipelineState_);
    encoder->setBuffer(params, 0, 0);
    encoder->setBuffer(bufferGradients_, 0, 1); // use internal gradient buffer
    encoder->setBuffer(bufferM_, 0, 2);
    encoder->setBuffer(bufferV_, 0, 3);
    encoder->setBytes(&timestep_, sizeof(uint32_t), 4);
    encoder->setBytes(&learningRate_, sizeof(float), 5);
    encoder->setBytes(&beta1_, sizeof(float), 6);
    encoder->setBytes(&beta2_, sizeof(float), 7);
    encoder->setBytes(&epsilon_, sizeof(float), 8);

    MTL::Size gridSize = MTL::Size(paramCount, 1, 1);
    NS::UInteger threadGroupSize = pipelineState_->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    encoder->dispatchThreads(gridSize, threadgroupSize);
}

MTL::Buffer* AdamOptimizer::gradientBuffer() const {
    return bufferGradients_;
}
