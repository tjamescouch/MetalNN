#include "dense-layer.h"
#include "common.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include "neural-engine.h"
#include "data-source.h"
#include "multi-layer-kernels.h"
#include "keyboard-controller.h"
#include "logger.h"

// Using kernel names "forward_output_layer" and "learn_output_layer".

DenseLayer::DenseLayer(int inputDim, int outputDim)
    : inputDim_(inputDim), outputDim_(outputDim),
      bufferInput_(nullptr), bufferOutput_(nullptr), bufferWeights_(nullptr), bufferBias_(nullptr),
      bufferYhat_(nullptr), bufferError_(nullptr),
      forwardPipelineState_(nullptr), backwardPipelineState_(nullptr)
{
}

DenseLayer::~DenseLayer() {
    if(bufferOutput_) bufferOutput_->release();
    if(bufferWeights_) bufferWeights_->release();
    if(bufferBias_) bufferBias_->release();
    if(bufferYhat_) bufferYhat_->release();
    if(bufferError_) bufferError_->release();
    if(forwardPipelineState_) forwardPipelineState_->release();
    if(backwardPipelineState_) backwardPipelineState_->release();
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_output_layer", NS::UTF8StringEncoding));
    auto backwardFunc = library->newFunction(NS::String::string("learn_output_layer", NS::UTF8StringEncoding));
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    if (!forwardPipelineState_) {
        std::cerr << "Error building forward pipeline state for DenseLayer: "
                  << error->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    if (!backwardPipelineState_) {
        std::cerr << "Error building backward pipeline state for DenseLayer: "
                  << error->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }
    forwardFunc->release();
    backwardFunc->release();
}

void DenseLayer::buildBuffers(MTL::Device* device) {
    // Allocate output buffer.
    bufferOutput_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferOutput_->contents(), 0, outputDim_ * sizeof(float));
    bufferOutput_->didModifyRange(NS::Range::Make(0, bufferOutput_->length()));
    
    // Allocate weight matrix (inputDim_ x outputDim_).
    bufferWeights_ = device->newBuffer(inputDim_ * outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    for (int i = 0; i < inputDim_ * outputDim_; i++) {
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    }
    bufferWeights_->didModifyRange(NS::Range::Make(0, bufferWeights_->length()));
    
    // Allocate bias vector.
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    memset(b, 0, outputDim_ * sizeof(float));
    bufferBias_->didModifyRange(NS::Range::Make(0, bufferBias_->length()));
    
    // Allocate target output buffer (y_hat) and error buffer for learning.
    bufferYhat_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferYhat_->contents(), 0, outputDim_ * sizeof(float));
    bufferYhat_->didModifyRange(NS::Range::Make(0, bufferYhat_->length()));
    
    bufferError_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferError_->contents(), 0, outputDim_ * sizeof(float));
    bufferError_->didModifyRange(NS::Range::Make(0, bufferError_->length()));
}

void DenseLayer::setInputBuffer(MTL::Buffer* inputBuffer) {
    bufferInput_ = inputBuffer;
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    // Bind: 0: input, 1: output, 2: weights, 3: bias, 4: hiddenDim (inputDim_), 5: outputDim.
    encoder->setBuffer(bufferInput_, 0, 0);
    encoder->setBuffer(bufferOutput_, 0, 1);
    encoder->setBuffer(bufferWeights_, 0, 2);
    encoder->setBuffer(bufferBias_, 0, 3);
    encoder->setBytes(&inputDim_, sizeof(int), 4);
    encoder->setBytes(&outputDim_, sizeof(int), 5);
    uint32_t threads = outputDim_;
    MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
    MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    encoder->endEncoding();
}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    // Bind for learn_output_layer:
    // 0: h (input buffer), 1: W, 2: b, 3: y (output buffer),
    // 4: y_hat (target), 5: error, 6: hiddenDim (inputDim_), 7: outputDim.
    encoder->setBuffer(bufferInput_, 0, 0);
    encoder->setBuffer(bufferWeights_, 0, 1);
    encoder->setBuffer(bufferBias_, 0, 2);
    encoder->setBuffer(bufferOutput_, 0, 3);
    encoder->setBuffer(bufferYhat_, 0, 4);
    encoder->setBuffer(bufferError_, 0, 5);
    encoder->setBytes(&inputDim_, sizeof(int), 6);
    encoder->setBytes(&outputDim_, sizeof(int), 7);
    uint32_t threads = outputDim_;
    MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
    MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    encoder->endEncoding();
}

MTL::Buffer* DenseLayer::getErrorBuffer() const {
    return bufferError_;
}

MTL::Buffer* DenseLayer::getOutputBuffer() const {
    return bufferOutput_;
}

void DenseLayer::updateTargetBuffer(DataSource& ds) {
    // Copy the target data (y_hat) from the DataSource into the Metal buffer.
    memcpy(bufferYhat_->contents(), ds.get_data_buffer(), ds.get_num_data() * sizeof(float));
    bufferYhat_->didModifyRange(NS::Range::Make(0, bufferYhat_->length()));
}
