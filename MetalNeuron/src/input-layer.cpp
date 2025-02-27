//
//  input-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#include "input-layer.h"
#include "common.h"  // For NS::Range
#include <cstring>   // For memcpy

InputLayer::InputLayer(int inputDim)
    : inputDim_(inputDim), bufferInput_(nullptr)
{
}

InputLayer::~InputLayer() {
    if(bufferInput_) {
        bufferInput_->release();
    }
}

void InputLayer::buildBuffers(MTL::Device* device) {
    // Allocate a buffer to hold 'inputDim_' float elements.
    bufferInput_ = device->newBuffer(inputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    // Initialize the buffer to zero.
    memset(bufferInput_->contents(), 0, inputDim_ * sizeof(float));
    bufferInput_->didModifyRange(NS::Range::Make(0, bufferInput_->length()));
}

void InputLayer::updateBuffer(DataSource& ds) {
    memcpy(bufferInput_->contents(), ds.get_data_buffer(), ds.get_num_data() * sizeof(float));
    bufferInput_->didModifyRange(NS::Range::Make(0, bufferInput_->length()));
}

MTL::Buffer* InputLayer::getBuffer() const {
    return bufferInput_;
}
