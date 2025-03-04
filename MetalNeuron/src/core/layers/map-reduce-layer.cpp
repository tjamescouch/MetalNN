//
//  map-reduce-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#include "map-reduce-layer.h"
#include <stdexcept>

MapReduceLayer::MapReduceLayer(int inputSize, int sequenceLength, ReductionType reductionType)
: inputSize_(inputSize),
  sequenceLength_(sequenceLength),
  reductionType_(reductionType),
  forwardPipelineState_(nullptr),
  backwardPipelineState_(nullptr)
{
    output_dim_ = inputSize_; // fixed-size output equal to input feature size
    sequenceLength_ = sequenceLength;
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

int MapReduceLayer::getSequenceLength() const {
    return 1; // Output is fixed size after reduction
}
