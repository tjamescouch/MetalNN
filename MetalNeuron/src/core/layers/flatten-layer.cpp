//
//  reshape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "flatten-layer.h"
#include "logger.h"

FlattenLayer::FlattenLayer(int sequenceLength, int featureDim, int batchSize) :
    sequenceLength_(sequenceLength),
    featureDim_(featureDim),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {
    assert(outputSize() == sequenceLength_ * featureDim_ &&
           "FlattenLayer output size mismatch: outputSize must equal sequenceLength * featureDim");
}

FlattenLayer::~FlattenLayer() {}

void FlattenLayer::buildBuffers(MTL::Device* device) {
    // Explicitly no buffer allocation required, reusing buffers from connected layers
}

void FlattenLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Explicitly no compute pipeline needed for FlattenLayer
}

void FlattenLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    /* FIXME - bind buffers */

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void FlattenLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    /* FIXME - bind buffers */

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void FlattenLayer::setInputBufferAt(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* FlattenLayer::getOutputBufferAt(BufferType type) { return outputBuffers_[type]; }
void FlattenLayer::setOutputBufferAt(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* FlattenLayer::getInputBufferAt(BufferType type) { return inputBuffers_[type]; }

int FlattenLayer::inputSize() const { return featureDim_; }
int FlattenLayer::outputSize() const { return featureDim_; }

void FlattenLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "FlattenLayer cannot be used as a terminal layer with targets.");
}

void FlattenLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "FlattenLayer cannot be used as a terminal layer with targets.");
}

void FlattenLayer::connectForwardConnections(Layer* previousLayer) {
    assert(previousLayer->outputSize() == sequenceLength_ * featureDim_ &&
       "FlattenLayer input size mismatch: previous layer output size must equal sequenceLength * featureDim");
    setInputBufferAt(BufferType::Input, previousLayer->getOutputBufferAt(BufferType::Output));
    setOutputBufferAt(BufferType::Output, this->getInputBufferAt(BufferType::Input));
}

void FlattenLayer::connectBackwardConnections(Layer* prevLayer)
{
    setOutputBufferAt(BufferType::OutgoingErrors, this->getInputBufferAt(BufferType::IncomingErrors));
    prevLayer->setInputBufferAt(BufferType::IncomingErrors, getOutputBufferAt(BufferType::OutgoingErrors));
}

void FlattenLayer::debugLog() {}

void FlattenLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void FlattenLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void FlattenLayer::saveParameters(std::ostream&) const {}
void FlattenLayer::loadParameters(std::istream&) {}

void FlattenLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
