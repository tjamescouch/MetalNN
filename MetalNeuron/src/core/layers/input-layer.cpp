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

InputLayer::InputLayer(int sequenceLength, int inputDim, int batchSize) :
sequenceLength_(sequenceLength),
inputDim_(inputDim),
isTerminal_(false),
batchSize_(batchSize)
{
    outputBuffers_[BufferType::Output].resize(1, nullptr);
    assert(outputBuffers_[BufferType::Output].size() > 0);
    Logger::log << "Constructor: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
}

InputLayer::~InputLayer() {
    for (auto ob : outputBuffers_) {
        ob.second[0]->release();
    }
}

void InputLayer::buildBuffers(MTL::Device* device) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    
    size_t bufferSize = batchSize_ * sequenceLength_ * inputDim_ * sizeof(float);
    outputBuffers_[BufferType::Output][0] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    // Initialize buffer content to zeros.
    memset(outputBuffers_[BufferType::Output][0]->contents(), 0, bufferSize);
    outputBuffers_[BufferType::Output][0]->didModifyRange(NS::Range::Make(0, bufferSize));
    
    assert(outputBuffers_[BufferType::Output].size() > 0);
    Logger::log << "buildBuffers: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
}

void InputLayer::updateBufferAt(const float* data) {
    Logger::log << "updateBufferAt: bufer output ptr: " << outputBuffers_[BufferType::Output][0] << std::endl;
    
    assert(outputBuffers_[BufferType::Output].size() > 0);
    assert(outputBuffers_[BufferType::Output][0]!=nullptr);
    
    memcpy(outputBuffers_[BufferType::Output][0]->contents(),
           data,
           sequenceLength_ * inputDim_ * batchSize_ * sizeof(float));
    outputBuffers_[BufferType::Output][0]->didModifyRange(NS::Range::Make(0, outputBuffers_[BufferType::Output][0]->length()));
}

void InputLayer::updateBufferAt(const float* data, int batchSize) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    
    memcpy(outputBuffers_[BufferType::Output][0]->contents(), data, sequenceLength_ * inputDim_ * batchSize * sizeof(float));
    outputBuffers_[BufferType::Output][0]->didModifyRange(NS::Range::Make(0, outputBuffers_[BufferType::Output][0]->length()));
}

void InputLayer::setInputBufferAt(BufferType type, MTL::Buffer* buffer) {
    // Intentionally empty
}

MTL::Buffer* InputLayer::getOutputBufferAt(BufferType type) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    return outputBuffers_[type][0];
}

void InputLayer::setOutputBufferAt(BufferType type, MTL::Buffer* buffer) {
    assert(outputBuffers_[BufferType::Output].size() > 0);
    outputBuffers_[type][0] = buffer;
    assert(outputBuffers_[BufferType::Output].size() > 0);
}

void InputLayer::resetErrors() {
}

MTL::Buffer* InputLayer::getInputBufferAt(BufferType) {
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
