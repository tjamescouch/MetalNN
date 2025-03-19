//
//  rehape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "reshape-layer.h"
#include "logger.h"

ReshapeLayer::ReshapeLayer(int featureDim, int batchSize) :
    featureDim_(featureDim),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {}

ReshapeLayer::~ReshapeLayer() {}

void ReshapeLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sizeof(float); //FIXME
    
    outputBuffers_[BufferType::Output] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

}

void ReshapeLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFn = library->newFunction(NS::String::string("forward_flatten", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_flatten", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);
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
void FlattenLayer::setOutputBufferAt(BufferType type, MTL::Buffer* buffer) {
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
    setInputBufferAt(BufferType::Input, previousLayer->getOutputBufferAt(BufferType::Output));
}

void ReshapeLayer::connectBackwardConnections(Layer* prevLayer)
{
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
