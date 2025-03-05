//
//  dropout-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//
#include <iostream>
#include <random>

#include "input-layer.h"
#include "dropout-layer.h"
#include "training-manager.h"

DropoutLayer::DropoutLayer(float rate, int featureDim, int sequenceLength)
: rate_(rate), featureDim_(featureDim), sequenceLength_(sequenceLength), bufferRandomMask_(nullptr),forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr), isTerminal_(false) {
    inputBuffers_[BufferType::Input].resize(0, nullptr);
    outputBuffers_[BufferType::Output].resize(0, nullptr);
}

DropoutLayer::~DropoutLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if(bufferRandomMask_) bufferRandomMask_->release();
    
    if(forwardPipelineState_) forwardPipelineState_->release();
    if(backwardPipelineState_) backwardPipelineState_->release();
}

void DropoutLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
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

void DropoutLayer::buildBuffers(MTL::Device* device) {
    assert(device && "Device is null!");
    

    inputBuffers_[BufferType::Input].clear();
    outputBuffers_[BufferType::Output].clear();
    inputBuffers_[BufferType::InputErrors].clear();
    outputBuffers_[BufferType::OutputErrors].clear();

    for(int t = 0; t < sequenceLength_; ++t) {
        auto inputBuf = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        assert(inputBuf && "Failed to allocate input buffer");
        inputBuffers_[BufferType::Input].push_back(inputBuf);

        auto outputBuf = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        assert(outputBuf && "Failed to allocate output buffer");
        outputBuffers_[BufferType::Output].push_back(outputBuf);

        auto inputErrBuf = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        assert(inputErrBuf && "Failed to allocate input error buffer");
        inputBuffers_[BufferType::InputErrors].push_back(inputErrBuf);

        auto outputErrBuf = device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        assert(outputErrBuf && "Failed to allocate output error buffer");
        outputBuffers_[BufferType::OutputErrors].push_back(outputErrBuf);
    }

    generateRandomMask();
    assert(bufferRandomMask_ && "Random mask buffer allocation failed");
}


void DropoutLayer::forward(MTL::CommandBuffer* cmdBuf) {
    bool isTraining = TrainingManager::instance().isTraining();
    
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 1);
        encoder->setBuffer(bufferRandomMask_, 0, 2);
        encoder->setBytes(&rate_, sizeof(float), 3);
        encoder->setBytes(&featureDim_, sizeof(int), 4);
        encoder->setBytes(&isTraining, sizeof(bool), 5);

        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
        
        inputBuffers_[BufferType::Input][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][t]->length()));
    }
}

void DropoutLayer::backward(MTL::CommandBuffer* cmdBuf) {
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::InputErrors][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 1);
        encoder->setBuffer(bufferRandomMask_, 0, 2);
        encoder->setBytes(&rate_, sizeof(float), 3);
        encoder->setBytes(&featureDim_, sizeof(int), 4);

        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void DropoutLayer::generateRandomMask() {
    std::vector<float> maskData(featureDim_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : maskData) {
        val = dist(gen);
    }

    if (bufferRandomMask_) bufferRandomMask_->release();
    
    bufferRandomMask_ = _pDevice->newBuffer(maskData.data(), featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
}

void DropoutLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* DropoutLayer::getOutputBufferAt(BufferType type, int timestep) {
    return outputBuffers_[BufferType::Output][timestep];
}

void DropoutLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* DropoutLayer::getInputBufferAt(BufferType type, int timestep) {
    return inputBuffers_[type][timestep];
}

void DropoutLayer::connectInputBuffers(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

int DropoutLayer::getParameterCount() const {
    return 1;
}

float DropoutLayer::getParameterAt(int index) const {
    return 0.0f;
}

void DropoutLayer::setParameterAt(int index, float value) {
    return;
}

float DropoutLayer::getGradientAt(int index) const {
    return 0.0f;
}

void DropoutLayer::saveParameters(std::ostream& os) const {
    // No parameters to save
}

void DropoutLayer::loadParameters(std::istream& is) {
    // No parameters to load
}
