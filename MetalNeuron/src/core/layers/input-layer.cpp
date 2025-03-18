//
//  input-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//
#include <cstring>   // For memcpy
#include <vector>
#include <iostream>

#include "input-layer.h"
#include "logger.h"
#include "common.h"  // For NS::Range

InputLayer::InputLayer(int inputDim, int sequenceLength, int batchSize)
: inputDim_(inputDim), sequenceLength_(sequenceLength),
  isTerminal_(false), batchSize_(batchSize)
{
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    assert(outputBuffers_[BufferType::Output].size() > 0);
    Logger::log << "Constructor: bufer output size in timesteps: " <<outputBuffers_[BufferType::Output].size() << std::endl;
    Logger::log << "Constructor: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
}

InputLayer::~InputLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
}

void InputLayer::buildBuffers(MTL::Device* device) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    // Allocate buffers for each timestep in the sequence.
    for (int t = 0; t < sequenceLength_; ++t) {
        outputBuffers_[BufferType::Output][t] = device->newBuffer(inputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
        // Initialize buffer content to zeros.
        memset(outputBuffers_[BufferType::Output][t]->contents(), 0, inputDim_ * batchSize_ * sizeof(float));
        outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range::Make(0, inputDim_ * batchSize_ * sizeof(float)));
    }
    assert(outputBuffers_[BufferType::Output].size() > 0);
    Logger::log << "buildBuffers: bufer output size in timesteps: " <<outputBuffers_[BufferType::Output].size() << std::endl;
    Logger::log << "buildBuffers: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
}

void InputLayer::updateBufferAt(const float* data, int timestep) {
    Logger::log << "updateBufferAt: bufer output size in timesteps: " <<outputBuffers_[BufferType::Output].size() << std::endl;
    Logger::log << "updateBufferAt: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
    
    assert(timestep >= 0  && timestep < sequenceLength_);
    assert(outputBuffers_[BufferType::Output].size() > 0);
    assert(outputBuffers_[BufferType::Output][0]!=nullptr);
    
    memcpy(outputBuffers_[BufferType::Output][0]->contents(),
           data,
           inputDim_ * batchSize_ * sizeof(float));
    outputBuffers_[BufferType::Output][0]->didModifyRange(NS::Range::Make(0, outputBuffers_[BufferType::Output][0]->length()));
}

void InputLayer::updateBufferAt(const float* data, int timestep, int batchSize) {
    assert(timestep == 0);
    assert(outputBuffers_[BufferType::Output].size() > 0);
    
    memcpy(outputBuffers_[BufferType::Output][timestep]->contents(), data, inputDim_ * batchSize * sizeof(float));
    outputBuffers_[BufferType::Output][timestep]->didModifyRange(NS::Range::Make(0, outputBuffers_[BufferType::Output][timestep]->length()));
}

void InputLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* InputLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    return outputBuffers_[type][timestep];
}

void InputLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    outputBuffers_[type][timestep] = buffer;
    assert(outputBuffers_[BufferType::Output].size() > 0);
}

MTL::Buffer* InputLayer::getInputBufferAt(BufferType, int) {
    return nullptr; // Input layer doesn't propagate error backwards
}

void InputLayer::saveParameters(std::ostream& os) const {
    // No parameters to save
}

void InputLayer::loadParameters(std::istream& is) {
    // No parameters to load
}

void InputLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output][0], getName());
}

void InputLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    Logger::instance().assertBufferContentsAreValid(outputBuffers_[BufferType::Output][0], getName());
}
