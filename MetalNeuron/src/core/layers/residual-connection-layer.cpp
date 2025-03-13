//
//  residual-connection-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "residual-connection-layer.h"

ResidualConnectionLayer::ResidualConnectionLayer(int featureDim, int batchSize)
    : featureDim_(featureDim), batchSize_(batchSize), isTerminal_(false),
      residualInputBuffer_(nullptr),
      forwardPipelineState_(nullptr), backwardPipelineState_(nullptr) {}

ResidualConnectionLayer::~ResidualConnectionLayer() {}

void ResidualConnectionLayer::setResidualInput(MTL::Buffer* residualBuffer) {
    residualInputBuffer_ = residualBuffer;
}

void ResidualConnectionLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sizeof(float);
    inputBuffers_.push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    outputBuffers_.push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
}

void ResidualConnectionLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFn = library->newFunction(NS::String::string("forward_residual", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_residual", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);
}

void ResidualConnectionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    encoder->setBuffer(inputBuffers_[0], 0, 0);
    encoder->setBuffer(residualInputBuffer_, 0, 1);
    encoder->setBuffer(outputBuffers_[0], 0, 2);

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ResidualConnectionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    encoder->setBuffer(inputBuffers_[0], 0, 0);
    encoder->setBuffer(outputBuffers_[0], 0, 1);
    encoder->setBuffer(residualInputBuffer_, 0, 2);

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

// Placeholder methods (implement as necessary)
void ResidualConnectionLayer::setInputBufferAt(BufferType, int, MTL::Buffer*) {}
MTL::Buffer* ResidualConnectionLayer::getOutputBufferAt(BufferType, int) { return outputBuffers_[0]; }
void ResidualConnectionLayer::setOutputBufferAt(BufferType, int, MTL::Buffer*) {}
MTL::Buffer* ResidualConnectionLayer::getInputBufferAt(BufferType, int) { return inputBuffers_[0]; }

int ResidualConnectionLayer::inputSize() const { return featureDim_; }
int ResidualConnectionLayer::outputSize() const { return featureDim_; }

void ResidualConnectionLayer::updateTargetBufferAt(const float*, int) {}
void ResidualConnectionLayer::updateTargetBufferAt(const float*, int, int) {}

void ResidualConnectionLayer::connectForwardConnections(Layer*, Layer*, MTL::Buffer*, int) {}
void ResidualConnectionLayer::connectBackwardConnections(Layer*, Layer*, MTL::Buffer*, int) {}

void ResidualConnectionLayer::debugLog() {}
void ResidualConnectionLayer::onForwardComplete(MTL::CommandQueue*, int) {}
void ResidualConnectionLayer::onBackwardComplete(MTL::CommandQueue*, int) {}

void ResidualConnectionLayer::saveParameters(std::ostream&) const {}
void ResidualConnectionLayer::loadParameters(std::istream&) {}

void ResidualConnectionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
