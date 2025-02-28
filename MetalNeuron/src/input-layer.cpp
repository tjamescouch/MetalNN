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

int globaltime = 0;

InputLayer::InputLayer(int inputDim, int sequenceLength)
    : inputDim_(inputDim), sequenceLength_(sequenceLength)
{
    bufferInputs_.resize(sequenceLength_, nullptr);
}

InputLayer::~InputLayer() {
    for (auto buffer : bufferInputs_) {
        if (buffer) {
            buffer->release();
        }
    }
}

void InputLayer::buildBuffers(MTL::Device* device) {
    // Allocate buffers for each timestep in the sequence.
    for (int t = 0; t < sequenceLength_; ++t) {
        bufferInputs_[t] = device->newBuffer(inputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        // Initialize buffer content to zeros.
        memset(bufferInputs_[t]->contents(), 0, inputDim_ * sizeof(float));
        bufferInputs_[t]->didModifyRange(NS::Range::Make(0, inputDim_ * sizeof(float)));
    }
}

void InputLayer::updateBufferAt(DataSource& ds, int timestep) {
    if (timestep < 0 || timestep >= sequenceLength_) {
        // Handle invalid timestep gracefully.
        return;
    }
    memcpy(bufferInputs_[timestep]->contents(),
           ds.get_data_buffer_at(globaltime++),
           inputDim_ * sizeof(float));
    bufferInputs_[timestep]->didModifyRange(NS::Range::Make(0, inputDim_ * sizeof(float)));
}

MTL::Buffer* InputLayer::getBufferAt(int timestep) const {
    if (timestep < 0 || timestep >= sequenceLength_) {
        return nullptr;  // Handle invalid timestep gracefully.
    }
    return bufferInputs_[timestep];
}
