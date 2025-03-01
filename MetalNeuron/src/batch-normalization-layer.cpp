//
//  batch-normalization-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "batch-normalization-layer.h"
#include <cassert>
#include <random>
#include <cstring>
#include <iostream>

BatchNormalizationLayer::BatchNormalizationLayer(int featureDim, int sequenceLength, float epsilon)
    : featureDim_(featureDim),
      sequenceLength_(sequenceLength),
      epsilon_(epsilon),
      bufferGamma_(nullptr),
      bufferBeta_(nullptr),
      bufferRunningMean_(nullptr),
      bufferRunningVariance_(nullptr),
      forwardPipelineState_(nullptr),
      backwardPipelineState_(nullptr) {}

BatchNormalizationLayer::~BatchNormalizationLayer() {
    if (bufferGamma_) bufferGamma_->release();
    if (bufferBeta_) bufferBeta_->release();
    if (bufferRunningMean_) bufferRunningMean_->release();
    if (bufferRunningVariance_) bufferRunningVariance_->release();

    for (auto buf : bufferInputs_) buf->release();
    for (auto buf : bufferOutputs_) buf->release();
    for (auto buf : bufferInputErrors_) buf->release();
    for (auto buf : bufferOutputErrors_) buf->release();

    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void BatchNormalizationLayer::initializeParameters(MTL::Device* device) {
    std::vector<float> gamma(featureDim_, 1.0f);
    std::vector<float> beta(featureDim_, 0.0f);
    std::vector<float> runningMean(featureDim_, 0.0f);
    std::vector<float> runningVariance(featureDim_, 1.0f);

    bufferGamma_ = device->newBuffer(gamma.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeShared);
    bufferBeta_ = device->newBuffer(beta.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeShared);
    bufferRunningMean_ = device->newBuffer(runningMean.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeShared);
    bufferRunningVariance_ = device->newBuffer(runningVariance.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeShared);
}

void BatchNormalizationLayer::buildBuffers(MTL::Device* device) {
    initializeParameters(device);

    bufferInputs_.clear();
    bufferOutputs_.clear();
    bufferInputErrors_.clear();
    bufferOutputErrors_.clear();

    for(int t = 0; t < sequenceLength_; ++t) {
        bufferInputs_.push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared));
        bufferOutputs_.push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared));
        bufferInputErrors_.push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared));
        bufferOutputErrors_.push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared));
    }
}

void BatchNormalizationLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto forwardFunction = library->newFunction(NS::String::string("forward_batch_norm", NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        std::cerr << "Forward pipeline error (BatchNorm): "
                  << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(forwardPipelineState_);
    forwardFunction->release();

    auto backwardFunction = library->newFunction(NS::String::string("backward_batch_norm", NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        std::cerr << "Backward pipeline error (BatchNorm): "
                  << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(backwardPipelineState_);
    backwardFunction->release();
}

void BatchNormalizationLayer::forward(MTL::CommandBuffer* cmdBuf) {
    bool isTraining = true; // Set accordingly or maintain a class-level state if dynamic.

    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);

        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer(bufferOutputs_[t], 0, 1);
        encoder->setBuffer(bufferGamma_, 0, 2);
        encoder->setBuffer(bufferBeta_, 0, 3);
        encoder->setBuffer(bufferRunningMean_, 0, 4);
        encoder->setBuffer(bufferRunningVariance_, 0, 5);
        encoder->setBytes(&epsilon_, sizeof(float), 6);
        encoder->setBytes(&featureDim_, sizeof(int), 7);
        encoder->setBytes(&isTraining, sizeof(bool), 8);

        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void BatchNormalizationLayer::backward(MTL::CommandBuffer* cmdBuf) {
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);

        encoder->setBuffer(bufferOutputs_[t], 0, 0);
        encoder->setBuffer(bufferOutputErrors_[t], 0, 1);
        encoder->setBuffer(bufferInputErrors_[t], 0, 2);
        encoder->setBuffer(bufferGamma_, 0, 3);
        encoder->setBuffer(bufferBeta_, 0, 4);
        encoder->setBytes(&epsilon_, sizeof(float), 5);
        encoder->setBytes(&featureDim_, sizeof(int), 6);

        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void BatchNormalizationLayer::setInputBufferAt(int timestep, MTL::Buffer* buffer) {
    bufferInputs_[timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getOutputBufferAt(int timestep) const {
    return bufferOutputs_[timestep];
}

void BatchNormalizationLayer::setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) {
    bufferOutputErrors_[timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getInputErrorBufferAt(int timestep) const {
    return bufferInputErrors_[timestep];
}

int BatchNormalizationLayer::outputSize() const {
    return featureDim_;
}

MTL::Buffer* BatchNormalizationLayer::getErrorBufferAt(int timestep) const {
    return bufferOutputErrors_[timestep];
}

void BatchNormalizationLayer::updateTargetBufferAt(DataSource& targetData, int timestep) {}
