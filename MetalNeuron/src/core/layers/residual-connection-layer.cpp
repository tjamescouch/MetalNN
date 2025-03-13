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

ResidualConnectionLayer* ResidualConnectionLayer::setResidualInput(MTL::Buffer* residualBuffer) {
    residualInputBuffer_ = residualBuffer;
    
    return this;
}

void ResidualConnectionLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sizeof(float);
    
    outputBuffers_[BufferType::Output] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutputErrors] = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    residualOutputErrorBuffer_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
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
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(residualInputBuffer_, 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 2);

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ResidualConnectionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    encoder->setBuffer(inputBuffers_[BufferType::InputErrors], 0, 0);   // input error coming from next layer
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors], 0, 1); // propagate back to previous layer
    encoder->setBuffer(residualOutputErrorBuffer_, 0, 2);               // propagate error back to residual source layer

    MTL::Size gridSize = MTL::Size(batchSize_ * featureDim_, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(std::min(1024, batchSize_ * featureDim_), 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);
    encoder->endEncoding();
}

void ResidualConnectionLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* ResidualConnectionLayer::getOutputBufferAt(BufferType type, int) { return outputBuffers_[type]; }
void ResidualConnectionLayer::setOutputBufferAt(BufferType type, int, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* ResidualConnectionLayer::getInputBufferAt(BufferType type, int) { return inputBuffers_[type]; }

int ResidualConnectionLayer::inputSize() const { return featureDim_; }
int ResidualConnectionLayer::outputSize() const { return featureDim_; }

void ResidualConnectionLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(false && "ResidualConnectionLayer cannot be used as a terminal layer with targets.");
}

void ResidualConnectionLayer::updateTargetBufferAt(const float* targetData, int timestep, int batchSize) {
    assert(false && "ResidualConnectionLayer cannot be used as a terminal layer with targets.");
}

void ResidualConnectionLayer::connectForwardConnections(Layer* previousLayer, Layer*, MTL::Buffer*, int) {
    inputBuffers_[BufferType::Input] = previousLayer->getOutputBufferAt(BufferType::Output, 0);
}

void ResidualConnectionLayer::connectBackwardConnections(Layer* previousLayer, Layer*, MTL::Buffer* zeroBuffer, int) {
    previousLayer->setOutputBufferAt(BufferType::OutputErrors, 0, inputBuffers_[BufferType::InputErrors]);
}

void ResidualConnectionLayer::debugLog() {}
void ResidualConnectionLayer::onForwardComplete(MTL::CommandQueue*, int) {}
void ResidualConnectionLayer::onBackwardComplete(MTL::CommandQueue*, int) {}

void ResidualConnectionLayer::saveParameters(std::ostream&) const {}
void ResidualConnectionLayer::loadParameters(std::istream&) {}

void ResidualConnectionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
