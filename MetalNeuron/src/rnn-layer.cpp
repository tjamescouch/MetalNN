#include "rnn-layer.h"
#include "common.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>

RNNLayer::RNNLayer(int inputDim, int hiddenDim)
    : inputDim_(inputDim), hiddenDim_(hiddenDim),
      bufferInput_(nullptr), bufferHidden_(nullptr), bufferHiddenPrev_(nullptr),
      bufferW_xh_(nullptr), bufferW_hh_(nullptr), bufferBias_(nullptr),
      bufferError_(nullptr), bufferDenseError_(nullptr),
      forwardPipelineState_(nullptr), backwardPipelineState_(nullptr)
{
}

RNNLayer::~RNNLayer() {
    if(bufferHidden_) bufferHidden_->release();
    if(bufferHiddenPrev_) bufferHiddenPrev_->release();
    if(bufferW_xh_) bufferW_xh_->release();
    if(bufferW_hh_) bufferW_hh_->release();
    if(bufferBias_) bufferBias_->release();
    if(bufferError_) bufferError_->release();
    if(forwardPipelineState_) forwardPipelineState_->release();
    if(backwardPipelineState_) backwardPipelineState_->release();
}

void RNNLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_rnn", NS::UTF8StringEncoding));
    auto backwardFunc = library->newFunction(NS::String::string("learn_rnn", NS::UTF8StringEncoding));

    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);

    forwardFunc->release();
    backwardFunc->release();
}

void RNNLayer::buildBuffers(MTL::Device* device) {
    bufferHidden_ = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferHidden_->contents(), 0, hiddenDim_ * sizeof(float));
    bufferHidden_->didModifyRange(NS::Range::Make(0, bufferHidden_->length()));

    bufferHiddenPrev_ = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferHiddenPrev_->contents(), 0, hiddenDim_ * sizeof(float));
    bufferHiddenPrev_->didModifyRange(NS::Range::Make(0, bufferHiddenPrev_->length()));

    bufferW_xh_ = device->newBuffer(inputDim_ * hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w_xh = static_cast<float*>(bufferW_xh_->contents());
    for (int i = 0; i < inputDim_ * hiddenDim_; i++)
        w_xh[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    bufferW_xh_->didModifyRange(NS::Range::Make(0, bufferW_xh_->length()));

    bufferW_hh_ = device->newBuffer(hiddenDim_ * hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w_hh = static_cast<float*>(bufferW_hh_->contents());
    for (int i = 0; i < hiddenDim_ * hiddenDim_; i++)
        w_hh[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    bufferW_hh_->didModifyRange(NS::Range::Make(0, bufferW_hh_->length()));

    bufferBias_ = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferBias_->contents(), 0, hiddenDim_ * sizeof(float));
    bufferBias_->didModifyRange(NS::Range::Make(0, bufferBias_->length()));

    bufferError_ = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferError_->contents(), 0, hiddenDim_ * sizeof(float));
    bufferError_->didModifyRange(NS::Range::Make(0, bufferError_->length()));
}

void RNNLayer::setInputBuffer(MTL::Buffer* inputBuffer) {
    bufferInput_ = inputBuffer;
}

void RNNLayer::setDenseErrorBuffer(MTL::Buffer* denseErrorBuffer) {
    bufferDenseError_ = denseErrorBuffer;
}

void RNNLayer::forward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(bufferInput_, 0, 0);
    encoder->setBuffer(bufferHiddenPrev_, 0, 1);
    encoder->setBuffer(bufferHidden_, 0, 2);
    encoder->setBuffer(bufferW_xh_, 0, 3);
    encoder->setBuffer(bufferW_hh_, 0, 4);
    encoder->setBuffer(bufferBias_, 0, 5);
    encoder->setBytes(&inputDim_, sizeof(int), 6);
    encoder->setBytes(&hiddenDim_, sizeof(int), 7);
    uint32_t threads = hiddenDim_;
    MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
    MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    encoder->endEncoding();
}

void RNNLayer::backward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    encoder->setBuffer(bufferInput_, 0, 0);
    encoder->setBuffer(bufferHiddenPrev_, 0, 1);
    encoder->setBuffer(bufferW_xh_, 0, 2);
    encoder->setBuffer(bufferW_hh_, 0, 3);
    encoder->setBuffer(bufferBias_, 0, 4);
    encoder->setBuffer(bufferHidden_, 0, 5);
    encoder->setBuffer(bufferDenseError_, 0, 6); // Critical binding fix!
    encoder->setBuffer(bufferError_, 0, 7);
    encoder->setBytes(&inputDim_, sizeof(int), 8);
    encoder->setBytes(&hiddenDim_, sizeof(int), 9);
    uint32_t threads = hiddenDim_;
    MTL::Size threadsPerThreadgroup = MTL::Size{ std::min(threads, 1024u), 1, 1 };
    MTL::Size threadgroups = MTL::Size{ (threads + 1023) / 1024, 1, 1 };
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    encoder->endEncoding();
}

MTL::Buffer* RNNLayer::getErrorBuffer() const {
    return bufferError_;
}

MTL::Buffer* RNNLayer::getOutputBuffer() const {
    return bufferHidden_;
}
