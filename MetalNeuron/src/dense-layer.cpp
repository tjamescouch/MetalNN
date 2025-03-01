#include "dense-layer.h"
#include "common.h"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

DenseLayer::DenseLayer(int inputDim, int outputDim, int sequenceLength, ActivationFunction activation)
: inputDim_(inputDim), outputDim_(outputDim), sequenceLength_(sequenceLength), activation_(activation),
bufferWeights_(nullptr), bufferBias_(nullptr), bufferDecay_(nullptr),
forwardPipelineState_(nullptr), backwardPipelineState_(nullptr)
{
    bufferInputs_.resize(sequenceLength_, nullptr);
    bufferOutputs_.resize(sequenceLength_, nullptr);
    bufferTargets_.resize(sequenceLength_, nullptr);
    bufferErrors_.resize(sequenceLength_, nullptr);
}

DenseLayer::~DenseLayer() {
    for (auto buf : bufferOutputs_) if (buf) buf->release();
    for (auto buf : bufferTargets_) if (buf) buf->release();
    for (auto buf : bufferErrors_)  if (buf) buf->release();
    
    if (bufferWeights_) bufferWeights_->release();
    if (bufferBias_) bufferBias_->release();
    if (bufferDecay_) bufferDecay_->release();
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_output_layer", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    auto backwardFunc = library->newFunction(NS::String::string("learn_output_layer", NS::UTF8StringEncoding));
    assert(backwardFunc && "Backward function not found.");
    
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);
    
    forwardFunc->release();
    backwardFunc->release();
}

void DenseLayer::buildBuffers(MTL::Device* device) {
    const float scale = 0.1f;
    const float decay = 1.0f;
    
    bufferWeights_ = device->newBuffer(inputDim_ * outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    for (int i = 0; i < inputDim_ * outputDim_; ++i)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    bufferDecay_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferDecay_->contents(), &decay, sizeof(float));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
    
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    memset(bufferBias_->contents(), 0, outputDim_ * sizeof(float));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    bufferInputErrors_.resize(sequenceLength_);
    bufferOutputErrors_.resize(sequenceLength_);
    
    for (int t = 0; t < sequenceLength_; ++t) {
        bufferInputErrors_[t] = device->newBuffer(inputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        bufferOutputErrors_[t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        
        bufferOutputs_[t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        bufferTargets_[t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        bufferErrors_[t]  = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        
        memset(bufferOutputs_[t]->contents(), 0, outputDim_ * sizeof(float));
        memset(bufferTargets_[t]->contents(), 0, outputDim_ * sizeof(float));
        memset(bufferErrors_[t]->contents(), 0, outputDim_ * sizeof(float));
        
        bufferOutputs_[t]->didModifyRange(NS::Range(0, bufferOutputs_[t]->length()));
        bufferTargets_[t]->didModifyRange(NS::Range(0, bufferTargets_[t]->length()));
        bufferErrors_[t]->didModifyRange(NS::Range(0, bufferErrors_[t]->length()));
    }
}

void DenseLayer::setInputBufferAt(int timestep, MTL::Buffer* inputBuffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferInputs_[timestep] = inputBuffer;
}

void DenseLayer::updateTargetBufferAt(DataSource& targetData, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    memcpy(bufferTargets_[timestep]->contents(), targetData.get_data_buffer_at(timestep), outputDim_ * sizeof(float));
    bufferTargets_[timestep]->didModifyRange(NS::Range(0, outputDim_ * sizeof(float)));
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);

    for (int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer(bufferOutputs_[t], 0, 1);
        encoder->setBuffer(bufferWeights_, 0, 2);
        encoder->setBuffer(bufferBias_, 0, 3);
        encoder->setBytes(&inputDim_, sizeof(int), 4);
        encoder->setBytes(&outputDim_, sizeof(int), 5);
        encoder->setBytes(&activationRaw, sizeof(uint), 6);

        MTL::Size threadsPerThreadgroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        encoder->endEncoding();
    }
}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);

    for (int t = sequenceLength_ - 1; t >= 0; --t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer(bufferWeights_, 0, 1);
        encoder->setBuffer(bufferBias_, 0, 2);
        encoder->setBuffer(bufferOutputs_[t], 0, 3);
        encoder->setBuffer(bufferTargets_[t], 0, 4);
        encoder->setBuffer(bufferErrors_[t], 0, 5);
        encoder->setBytes(&inputDim_, sizeof(int), 6);
        encoder->setBytes(&outputDim_, sizeof(int), 7);
        encoder->setBuffer(bufferDecay_, 0, 8);
        encoder->setBytes(&activationRaw, sizeof(uint), 9);  // Activation function passed here
        
        MTL::Size threadsPerThreadgroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        encoder->endEncoding();
    }
    
    // Ensure buffers modified by GPU kernels are synchronized
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
    for (int t = 0; t < sequenceLength_; ++t) {
        bufferErrors_[t]->didModifyRange(NS::Range(0, outputDim_ * sizeof(float)));
    }

#ifdef DEBUG_NETWORK
    float* weights = static_cast<float*>(bufferWeights_->contents());
    std::cout << "Weights sample after backward: "
    << weights[0] << ", " << weights[1] << ", ..." << std::endl;
#endif
}


MTL::Buffer* DenseLayer::getErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferErrors_[timestep];
}

MTL::Buffer* DenseLayer::getOutputBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferOutputs_[timestep];
}

int DenseLayer::outputSize() const {
    return outputDim_;
}

void DenseLayer::setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferOutputErrors_[timestep] = buffer;
}

MTL::Buffer* DenseLayer::getInputErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferInputErrors_[timestep];
}
