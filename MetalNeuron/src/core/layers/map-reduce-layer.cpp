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
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Delta].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_, nullptr);
}


MapReduceLayer::~MapReduceLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void MapReduceLayer::buildBuffers(MTL::Device* device) {
    inputBuffers_[BufferType::InputErrors].clear();
    inputBuffers_[BufferType::Input].clear();
    outputBuffers_[BufferType::Output].clear();
    outputBuffers_[BufferType::Delta].clear();
    outputBuffers_[BufferType::OutputErrors].clear();
    
    for (int t = 0; t < sequenceLength_; ++t) {
        inputBuffers_[BufferType::Input].push_back(
                                                   device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged)
                                                     );
        inputBuffers_[BufferType::InputErrors].push_back(
                                                   device->newBuffer(inputSize_ * sizeof(float), MTL::ResourceStorageModeManaged)
                                                     );
        outputBuffers_[BufferType::Output].push_back(
                                                     device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged)
                                                     );
        outputBuffers_[BufferType::Delta].push_back(
                                                    device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged)
                                                    );
        outputBuffers_[BufferType::OutputErrors].push_back(
                                                           device->newBuffer(output_dim_ * sizeof(float), MTL::ResourceStorageModeManaged)
                                                    );
    }
}

int MapReduceLayer::getSequenceLength() {
    return sequenceLength_;
}


void MapReduceLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
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

void MapReduceLayer::connectBackwardConnections(Layer* prevLayer,
                                   Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    prevLayer->setInputBufferAt(BufferType::InputErrors, 0, getOutputBufferAt(BufferType::OutputErrors, timestep));
}

void MapReduceLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto kernelNameForward = NS::String::string("forward_map_reduce", NS::UTF8StringEncoding);
    auto kernelNameBackward = NS::String::string("backward_map_reduce", NS::UTF8StringEncoding);
    
    MTL::Function* forwardFunction = library->newFunction(kernelNameForward);
    MTL::Function* backwardFunction = library->newFunction(kernelNameBackward);
    
    NS::Error* error = nullptr;
    
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        throw std::runtime_error("Error creating forward pipeline state");
    }
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        throw std::runtime_error("Error creating backward pipeline state");
    }
    
    forwardFunction->release();
    backwardFunction->release();
}

void MapReduceLayer::forward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);
    encoder->setBytes(&inputSize_, sizeof(int), 2);
    encoder->setBytes(&reductionType_, sizeof(int), 3);
    
    MTL::Size gridSize = MTL::Size(inputSize_, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(std::min(inputSize_, 1024), 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    
    encoder->endEncoding();
}

void MapReduceLayer::backward(MTL::CommandBuffer* cmdBuf) {
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    encoder->setBuffer(outputBuffers_[BufferType::Delta][0], 0, 0);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1); // forwardOutput buffer
    encoder->setBuffer(inputBuffers_[BufferType::InputErrors][0], 0, 2);
    encoder->setBytes(&inputSize_, sizeof(uint), 3);
    encoder->setBytes(&reductionType_, sizeof(uint), 4);
    
    MTL::Size gridSize = MTL::Size(inputSize_, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(fmin(inputSize_, 1024u), 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    
    encoder->endEncoding();
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
