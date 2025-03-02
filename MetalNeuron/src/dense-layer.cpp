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
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_, nullptr);
}

DenseLayer::~DenseLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ib : inputBuffers_) {
            ib.second[t]->release();
        }
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (bufferWeights_) bufferWeights_->release();
    if (bufferBias_) bufferBias_->release();
    if (bufferDecay_) bufferDecay_->release();
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_dense_layer", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    auto backwardFunc = library->newFunction(NS::String::string("learn_dense_layer", NS::UTF8StringEncoding));
    assert(backwardFunc && "Backward function not found.");
    
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);
    
    forwardFunc->release();
    backwardFunc->release();
}

#include "weight-initializer.h"

void DenseLayer::buildBuffers(MTL::Device* device) {
    const float decay = 1.0f;
    
    bufferWeights_ = device->newBuffer(inputDim_ * outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    WeightInitializer::initializeXavier(w, inputDim_, outputDim_);
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    WeightInitializer::initializeBias(b, outputDim_);
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));

    

    bufferDecay_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferDecay_->contents(), &decay, sizeof(float));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));

    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Delta].resize(sequenceLength_);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    
    
    for (int t = 0; t < sequenceLength_; ++t) {
        outputBuffers_[BufferType::Delta][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        outputBuffers_[BufferType::Delta][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Delta][t]->length()));
    }

    for (int t = 0; t < sequenceLength_; ++t) {
        
        outputBuffers_[BufferType::OutputErrors][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        outputBuffers_[BufferType::Output][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        
        outputBuffers_[BufferType::Debug][t]   = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        inputBuffers_[BufferType::Targets][t]  = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);

        memset(outputBuffers_[BufferType::Output][t]->contents(), 0, outputDim_ * sizeof(float));
        memset(inputBuffers_[BufferType::Targets][t]->contents(), 0, outputDim_ * sizeof(float));
        memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, outputDim_ * sizeof(float));
        memset(outputBuffers_[BufferType::Debug][t]->contents(), 0, outputDim_ * sizeof(float));

        inputBuffers_[BufferType::Targets][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][t]->length()));
        outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Output][t]->length()));
        outputBuffers_[BufferType::OutputErrors][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutputErrors][t]->length()));
        outputBuffers_[BufferType::Debug][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Debug][t]->length()));
    }
}
void DenseLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

void DenseLayer::updateTargetBufferAt(DataSource& targetData, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    memcpy(inputBuffers_[BufferType::Targets][timestep]->contents(), targetData.get_data_buffer_at(timestep), outputDim_ * sizeof(float));
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, outputDim_ * sizeof(float)));
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);

    for (int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 1);
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
        
        encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);         // Input activations (h)
        encoder->setBuffer(bufferWeights_, 0, 1);           // Weights (W)
        encoder->setBuffer(bufferBias_, 0, 2);              // Biases (b)
        encoder->setBuffer(inputBuffers_[BufferType::Targets][t], 0, 3);        // Target (y)
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 4);        // Predicted output (y_hat)
        encoder->setBuffer(outputBuffers_[BufferType::Delta][t], 0, 5);         // Error (delta) ??
        encoder->setBytes(&inputDim_, sizeof(uint), 6);     // Input dimension (pH)
        encoder->setBytes(&outputDim_, sizeof(uint), 7);    // Output dimension (pN)
        encoder->setBuffer(bufferDecay_, 0, 8);             // Decay factor (pDecay)
        encoder->setBytes(&activationRaw, sizeof(uint), 9); // Activation function type
        encoder->setBuffer(outputBuffers_[BufferType::Debug][t], 0, 10);  // debug signal
        encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 11);

        MTL::Size threadsPerThreadgroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        encoder->endEncoding();
    }

    // Sync buffer contents back to CPU (existing logic preserved)
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));

    for (int t = 0; t < sequenceLength_; ++t) {
        inputBuffers_[BufferType::InputErrors][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::InputErrors][t]->length()));
    }
}

void DenseLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* DenseLayer::getInputBufferAt(BufferType type, int timestep) const {
    auto it = inputBuffers_.find(type);
    if (it != inputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;  // Explicitly handle missing keys appropriately
}

MTL::Buffer* DenseLayer::getOutputBufferAt(BufferType type, int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    auto it = outputBuffers_.find(type);
    if (it != outputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;  // Explicitly handle the error scenario as needed
}

int DenseLayer::outputSize() const {
    return outputDim_;
}


void DenseLayer::connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
        previousLayer
            ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
            : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
    );
}
