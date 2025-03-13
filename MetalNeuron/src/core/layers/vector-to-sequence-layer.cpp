//
//  vector-to-sequence-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//
#include <iostream>
#include <random>

#include "logger.h"
#include "input-layer.h"
#include "vector-to-sequence-layer.h"
#include "training-manager.h"


VectorToSequenceLayer::VectorToSequenceLayer(int inputDim, int outputDim, int batchSize, int sequenceLength) :
featureDim_(inputDim * batchSize),
inputDim_(inputDim),
batchSize_(batchSize),
sequenceLength_(sequenceLength),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
isTerminal_(false) {
    assert(inputDim_ == outputDim);
    
    inputBuffers_[BufferType::Input].resize(1, nullptr);
    outputBuffers_[BufferType::Output].resize(1, nullptr);
    outputBuffers_[BufferType::Debug].resize(1, nullptr);
    
    inputBuffers_[BufferType::InputErrors].resize(1, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(1, nullptr);
}

VectorToSequenceLayer::~VectorToSequenceLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if(forwardPipelineState_) forwardPipelineState_->release();
    if(backwardPipelineState_) backwardPipelineState_->release();
}

void VectorToSequenceLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    _pDevice = device;
    
    auto forwardFunction = library->newFunction(NS::String::string("forward_dropout", NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        std::cerr << "Forward pipeline error (Dropout): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(forwardPipelineState_);
    forwardFunction->release();
    
    auto backwardFunction = library->newFunction(NS::String::string("backward_dropout", NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        std::cerr << "Backward pipeline error (Dropout): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(backwardPipelineState_);
    backwardFunction->release();
}

void VectorToSequenceLayer::buildBuffers(MTL::Device* device) {
    assert(device && "Device is null!");
    
    
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    
    int t = 0;
    

    outputBuffers_[BufferType::OutputErrors][t] = device->newBuffer(featureDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Output][t] = device->newBuffer(featureDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Debug][t] = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    memset(outputBuffers_[BufferType::Output][t]->contents(), 0, featureDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, featureDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::Debug][t]->contents(), 0, featureDim_ * sizeof(float));
    
    outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Output][t]->length()));
    outputBuffers_[BufferType::OutputErrors][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutputErrors][t]->length()));
    outputBuffers_[BufferType::Debug][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Debug][t]->length()));
    
    Logger::log << "dropout output error buffer initalized @" << outputBuffers_[BufferType::OutputErrors][t] << std::endl;
    
}


void VectorToSequenceLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    for(int t = 0; t < sequenceLength_; ++t) {
        /* No kernel code */
        
        //FIXME - implement
        
        inputBuffers_[BufferType::Input][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][t]->length()));
    }
}

void VectorToSequenceLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::InputErrors][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 1);
        encoder->setBytes(&featureDim_, sizeof(int), 2);
        encoder->setBuffer(outputBuffers_[BufferType::Debug][t], 0, 3);
        
        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void VectorToSequenceLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* VectorToSequenceLayer::getOutputBufferAt(BufferType type, int timestep) {
    return outputBuffers_[type][timestep];
}

void VectorToSequenceLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* VectorToSequenceLayer::getInputBufferAt(BufferType type, int timestep) {
    return inputBuffers_[type][timestep];
}

void VectorToSequenceLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

void VectorToSequenceLayer::connectBackwardConnections(Layer* prevLayer,
                                   Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    Logger::log << "dropout output error buffer @" << getOutputBufferAt(BufferType::OutputErrors, timestep) << std::endl;
    if (prevLayer) {
        prevLayer->setInputBufferAt(BufferType::InputErrors, timestep, getOutputBufferAt(BufferType::OutputErrors, timestep));
    }
}

void VectorToSequenceLayer::saveParameters(std::ostream& os) const {
    // No parameters to save
}

void VectorToSequenceLayer::loadParameters(std::istream& is) {
    // No parameters to load
}


void VectorToSequenceLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void VectorToSequenceLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}
