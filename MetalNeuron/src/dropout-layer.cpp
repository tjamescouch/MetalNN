//
//  dropout-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "dropout-layer.h"
#include <iostream>

DropoutLayer::DropoutLayer(float rate, int sequenceLength)
: rate_(rate), sequenceLength_(sequenceLength) {}

DropoutLayer::~DropoutLayer() {}

void DropoutLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Stub: we'll implement Metal pipeline next step
    std::cout << "⚙️ Dropout pipeline (stub) created with rate: " << rate_ << "\n";
}

void DropoutLayer::buildBuffers(MTL::Device* device) {
    // Stub: no buffers needed yet
    std::cout << "⚙️ Dropout buffers (stub) created\n";
}

void DropoutLayer::forward(MTL::CommandBuffer* cmdBuf) {
    std::cout << "🚀 DropoutLayer forward pass executed." << std::endl;
    // (Stub, to be implemented next increment)
}

void DropoutLayer::backward(MTL::CommandBuffer* cmdBuf) {
    // Stub: no-op for now
    std::cout << "🚀 DropoutLayer backward pass executed." << std::endl;
}
