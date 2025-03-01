//
//  dropout-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "dropout-layer.h"
#include <iostream>

DropoutLayer::DropoutLayer(float rate, int featureDim, int sequenceLength)
: rate_(rate), featureDim_(featureDim), sequenceLength_(sequenceLength) {}

DropoutLayer::~DropoutLayer() {}

void DropoutLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Stub: we'll implement Metal pipeline next step
    std::cout << "⚙️ Dropout pipeline (stub) created with rate: " << rate_ << "\n";
}

void DropoutLayer::buildBuffers(MTL::Device* device) {
    bufferInputs_.resize(sequenceLength_);
    bufferOutputs_.resize(sequenceLength_);
    bufferInputErrors_.resize(sequenceLength_);
    bufferOutputErrors_.resize(sequenceLength_);

    for (int t = 0; t < sequenceLength_; ++t) {
        bufferOutputs_[t] = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared);
        
        bufferInputErrors_[t] = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared);
        bufferOutputErrors_[t] = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeShared);
    }
}

void DropoutLayer::forward(MTL::CommandBuffer* cmdBuf) {
    for (int t = 0; t < sequenceLength_; ++t) {
        memcpy(bufferOutputs_[t]->contents(),
               bufferInputs_[t]->contents(),
               featureDim_ * sizeof(float));
    }
    std::cout << "🚀 DropoutLayer forward pass executed (stub copy)." << std::endl;
}

void DropoutLayer::backward(MTL::CommandBuffer* cmdBuf) {
    for (int t = 0; t < sequenceLength_; ++t) {
        memcpy(bufferInputErrors_[t]->contents(),
               bufferOutputErrors_[t]->contents(),
               featureDim_ * sizeof(float));
    }
    std::cout << "🚀 DropoutLayer backward pass executed (stub copy)." << std::endl;
}

void DropoutLayer::setInputBufferAt(int timestep, MTL::Buffer* buffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferInputs_[timestep] = buffer;
}

MTL::Buffer* DropoutLayer::getOutputBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    // During inference or after applying dropout, output buffer is returned:
    return bufferOutputs_[timestep];
}

void DropoutLayer::setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferOutputErrors_[timestep] = buffer;
}

MTL::Buffer* DropoutLayer::getInputErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferInputErrors_[timestep];
}
