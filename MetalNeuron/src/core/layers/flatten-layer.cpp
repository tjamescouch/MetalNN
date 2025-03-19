//
//  reshape-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "flatten-layer.h"
#include "logger.h"

FlattenLayer::FlattenLayer(int featureDim, int batchSize) :
    featureDim_(featureDim),
    batchSize_(batchSize),
    isTerminal_(false),
    forwardPipelineState_(nullptr),
    backwardPipelineState_(nullptr) {}

FlattenLayer::~FlattenLayer() {}

void FlattenLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sizeof(float); //FIXME
    
    outputBuffers_[BufferType::Output] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutgoingErrors] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

}

void FlattenLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
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
    setInputBufferAt(BufferType::Input, previousLayer->getOutputBufferAt(BufferType::Output));
}

void FlattenLayer::connectBackwardConnections(Layer* prevLayer)
{
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
