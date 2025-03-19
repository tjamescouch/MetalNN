//
//  rehape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "reshape-layer.h"
#include "logger.h"

ReshapeLayer::ReshapeLayer(int sequenceLength, int featureDim, int batchSize) :
    sequenceLength_(sequenceLength),
    featureDim_(featureDim),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {
    assert(outputSize() == sequenceLength_ * featureDim_ &&
           "ReshapeLayer output size mismatch: outputSize must equal sequenceLength * featureDim");
}

ReshapeLayer::~ReshapeLayer() {}

void ReshapeLayer::buildBuffers(MTL::Device* device) {
    // Explicitly no buffer allocation required, reusing buffers from connected layers
}

void ReshapeLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    // Explicitly no compute pipeline needed for ReshapeLayer
}

void ReshapeLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    /* FIXME - bind buffers */

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ReshapeLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    /* FIXME - bind buffers */

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ReshapeLayer::setInputBufferAt(BufferType type, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* ReshapeLayer::getOutputBufferAt(BufferType type) { return outputBuffers_[type]; }
void ReshapeLayer::setOutputBufferAt(BufferType type, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* ReshapeLayer::getInputBufferAt(BufferType type) { return inputBuffers_[type]; }

int ReshapeLayer::inputSize() const { return featureDim_; }
int ReshapeLayer::outputSize() const { return featureDim_; }

void ReshapeLayer::updateTargetBufferAt(const float* targetData) {
    assert(false && "ReshapeLayer cannot be used as a terminal layer with targets.");
}

void ReshapeLayer::updateTargetBufferAt(const float* targetData, int batchSize) {
    assert(false && "ReshapeLayer cannot be used as a terminal layer with targets.");
}

void ReshapeLayer::connectForwardConnections(Layer* previousLayer) {
    assert(previousLayer->outputSize() == sequenceLength_ * featureDim_ &&
       "ReshapeLayer input size mismatch: previous layer output size must equal sequenceLength * featureDim");
    setInputBufferAt(BufferType::Input, previousLayer->getOutputBufferAt(BufferType::Output));
    setOutputBufferAt(BufferType::Output, this->getInputBufferAt(BufferType::Input));
}

void ReshapeLayer::connectBackwardConnections(Layer* prevLayer)
{
    setOutputBufferAt(BufferType::OutgoingErrors, this->getInputBufferAt(BufferType::IncomingErrors));
    prevLayer->setInputBufferAt(BufferType::IncomingErrors, getOutputBufferAt(BufferType::OutgoingErrors));
}

void ReshapeLayer::debugLog() {}

void ReshapeLayer::onForwardComplete(MTL::CommandQueue*, int) {
}

void ReshapeLayer::onBackwardComplete(MTL::CommandQueue*, int) {
}

void ReshapeLayer::saveParameters(std::ostream&) const {}
void ReshapeLayer::loadParameters(std::istream&) {}

void ReshapeLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
