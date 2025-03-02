//
//  input-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#include "input-layer.h"
#include "common.h"  // For NS::Range
#include <cstring>   // For memcpy
#include <vector>

InputLayer::InputLayer(int inputDim, int sequenceLength)
    : inputDim_(inputDim), sequenceLength_(sequenceLength)
{
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
}

InputLayer::~InputLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ib : inputBuffers_) {
            ib.second[t]->release();
        }
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
}

void InputLayer::buildBuffers(MTL::Device* device) {
    // Allocate buffers for each timestep in the sequence.
    for (int t = 0; t < sequenceLength_; ++t) {
        outputBuffers_[BufferType::Output][t] = device->newBuffer(inputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        // Initialize buffer content to zeros.
        memset(outputBuffers_[BufferType::Output][t]->contents(), 0, inputDim_ * sizeof(float));
        outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range::Make(0, inputDim_ * sizeof(float)));
    }
}

void InputLayer::updateBufferAt(DataSource& ds, int timestep) {
    assert(timestep >= 0  && timestep < sequenceLength_);
    
    memcpy(outputBuffers_[BufferType::Output][timestep]->contents(),
           ds.get_data_buffer_at(timestep),
           inputDim_ * sizeof(float));
    outputBuffers_[BufferType::Output][timestep]->didModifyRange(NS::Range::Make(0, outputBuffers_[BufferType::Output][timestep]->length()));
}

void InputLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* InputLayer::getOutputBufferAt(BufferType type, int timestep) const {
    auto it = outputBuffers_.find(type);
    return (it != outputBuffers_.end()) ? it->second[timestep] : nullptr;
}

void InputLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* InputLayer::getInputBufferAt(BufferType, int) const {
    return nullptr; // Input layer doesn't propagate error backwards
}

int InputLayer::getSequenceLength() {
    return sequenceLength_;
}
