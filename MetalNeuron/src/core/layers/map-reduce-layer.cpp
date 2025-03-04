//
//  map-reduce-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//
#include "map-reduce-layer.h"
#include <stdexcept>
#include <iostream>


MapReduceLayer::MapReduceLayer(int inputSize, ReductionType reductionType)
: inputSize_(inputSize), output_dim_(1),
  sequenceLength_(1),
  reductionType_(reductionType),
  forwardPipelineState_(nullptr),
  backwardPipelineState_(nullptr),
  isTerminal_(false) {
}


MapReduceLayer::~MapReduceLayer() {
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void MapReduceLayer::buildBuffers(MTL::Device* device) {
    for (int t = 0; t < sequenceLength_; ++t) {
        outputBuffers_[BufferType::Output].push_back(
            device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged)
        );
        outputBuffers_[BufferType::Delta].push_back(
            device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged)
        );
    }

    // Input error buffer, for propagating errors back to previous layer
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    for (int t = 0; t < sequenceLength_; ++t) {
        inputBuffers_[BufferType::InputErrors][t] = device->newBuffer(
            inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged
        );
    }
}

int MapReduceLayer::getSequenceLength() {
    return sequenceLength_;
}


void MapReduceLayer::connectInputBuffers(Layer* previousLayer, Layer* inputLayer,
                                 MTL::Buffer* zeroBuffer, int timestep){
    if (previousLayer) {
        setInputBufferAt(BufferType::Input, timestep,
                         previousLayer->getOutputBufferAt(BufferType::Output, timestep));
    } else if (inputLayer) {
        setInputBufferAt(BufferType::Input, timestep,
                         inputLayer->getOutputBufferAt(BufferType::Output, timestep));
    } else {
        setInputBufferAt(BufferType::Input, timestep, zeroBuffer);
    }
}

void MapReduceLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Intentionally left empty for MapReduceLayer.
}

void MapReduceLayer::forward(MTL::CommandBuffer* cmdBuf) {
    // Intentionally left empty for MapReduceLayer.
}

void MapReduceLayer::backward(MTL::CommandBuffer* cmdBuf) {
    // Intentionally left empty for MapReduceLayer.
}


void MapReduceLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* MapReduceLayer::getOutputBufferAt(BufferType type, int timestep) {
    return outputBuffers_[type][timestep];
}

void MapReduceLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* MapReduceLayer::getInputBufferAt(BufferType type, int timestep) {
    return inputBuffers_[type][timestep];
}

int MapReduceLayer::outputSize() const {
    return output_dim_;
}

void MapReduceLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    // Typically a MapReduceLayer might ignore this or handle it differently
}

int MapReduceLayer::getParameterCount() const {
    return 0; // Adjust as needed
}

float MapReduceLayer::getParameterAt(int index) const {
    return 0; // Adjust as needed
}

void MapReduceLayer::setParameterAt(int index, float value) {
    // Implement as needed
}

float MapReduceLayer::getGradientAt(int index) const {
    return 0; // Adjust as needed
}

void MapReduceLayer::debugLog() {
    std::cout << "[MapReduceLayer] debugLog called." << std::endl;
}

void MapReduceLayer::onForwardComplete() {
    // Implement if necessary
}

void MapReduceLayer::onBackwardComplete(MTL::CommandQueue* queue) {
    // Implement if necessary
}

void MapReduceLayer::saveParameters(std::ostream& outStream) const {
    // Implement parameter serialization
}

void MapReduceLayer::loadParameters(std::istream& inStream) {
    // Implement parameter deserialization
}

void MapReduceLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
