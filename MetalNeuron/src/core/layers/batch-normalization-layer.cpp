//
//  batch-normalization-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "input-layer.h"
#include "batch-normalization-layer.h"
#include <cassert>
#include <random>
#include <cstring>
#include <iostream>
#include "training-manager.h"

BatchNormalizationLayer::BatchNormalizationLayer(int featureDim, int _unused, float epsilon)
: featureDim_(featureDim),
sequenceLength_(1),
epsilon_(epsilon),
bufferGamma_(nullptr),
bufferBeta_(nullptr),
bufferRunningMean_(nullptr),
bufferRunningVariance_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr) {
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
}

BatchNormalizationLayer::~BatchNormalizationLayer() {
    if (bufferGamma_) bufferGamma_->release();
    if (bufferBeta_) bufferBeta_->release();
    if (bufferRunningMean_) bufferRunningMean_->release();
    if (bufferRunningVariance_) bufferRunningVariance_->release();
    
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ib : inputBuffers_) {
            ib.second[t]->release();
        }
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void BatchNormalizationLayer::initializeParameters(MTL::Device* device) {
    std::vector<float> gamma(featureDim_, 1.0f);
    std::vector<float> beta(featureDim_, 0.0f);
    std::vector<float> runningMean(featureDim_, 0.0f);
    std::vector<float> runningVariance(featureDim_, 1.0f);
    
    bufferGamma_ = device->newBuffer(gamma.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(beta.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
    bufferRunningMean_ = device->newBuffer(runningMean.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
    bufferRunningVariance_ = device->newBuffer(runningVariance.data(), sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
}

void BatchNormalizationLayer::buildBuffers(MTL::Device* device) {
    initializeParameters(device);
    
    inputBuffers_[BufferType::Input].clear();
    outputBuffers_[BufferType::Output].clear();
    inputBuffers_[BufferType::InputErrors].clear();
    outputBuffers_[BufferType::OutputErrors].clear();
    
    inputBuffers_[BufferType::Input].push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::Output].push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged));
    inputBuffers_[BufferType::InputErrors].push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::OutputErrors].push_back(device->newBuffer(featureDim_ * sizeof(float), MTL::ResourceStorageModeManaged));
}

void BatchNormalizationLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFunction = library->newFunction(NS::String::string("forward_batch_norm", NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        std::cerr << "Forward pipeline error (BatchNorm): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(forwardPipelineState_);
    forwardFunction->release();
    
    auto backwardFunction = library->newFunction(NS::String::string("backward_batch_norm", NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        std::cerr << "Backward pipeline error (BatchNorm): "
        << error->localizedDescription()->utf8String() << std::endl;
    }
    assert(backwardPipelineState_);
    backwardFunction->release();
}

void BatchNormalizationLayer::forward(MTL::CommandBuffer* cmdBuf) {
    bool isTraining = TrainingManager::instance().isTraining();
    
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        
        encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 1);
        encoder->setBuffer(bufferGamma_, 0, 2);
        encoder->setBuffer(bufferBeta_, 0, 3);
        encoder->setBuffer(bufferRunningMean_, 0, 4);
        encoder->setBuffer(bufferRunningVariance_, 0, 5);
        encoder->setBytes(&epsilon_, sizeof(float), 6);
        encoder->setBytes(&featureDim_, sizeof(int), 7);
        encoder->setBytes(&isTraining, sizeof(bool), 8);
        
        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

void BatchNormalizationLayer::backward(MTL::CommandBuffer* cmdBuf) {
    for(int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 1);
        encoder->setBuffer(inputBuffers_[BufferType::InputErrors][t], 0, 2);
        encoder->setBuffer(bufferGamma_, 0, 3);
        encoder->setBuffer(bufferBeta_, 0, 4);
        encoder->setBytes(&epsilon_, sizeof(float), 5);
        encoder->setBytes(&featureDim_, sizeof(int), 6);
        
        MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
        encoder->endEncoding();
    }
}

int BatchNormalizationLayer::outputSize() const {
    return featureDim_;
}

void BatchNormalizationLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
}


void BatchNormalizationLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    auto it = outputBuffers_.find(type);
    if (it != outputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;
}

void BatchNormalizationLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getInputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    auto it = inputBuffers_.find(type);
    if (it != inputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;
}

void BatchNormalizationLayer::connectInputBuffers(Layer* previousLayer, InputLayer* inputLayer,
                                                  MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

int BatchNormalizationLayer::getParameterCount() const {
    return 2;
}
float BatchNormalizationLayer::getParameterAt(int index) const {
    return 0.0f;
}
void BatchNormalizationLayer::setParameterAt(int index, float value) {
    return;
}
float BatchNormalizationLayer::getGradientAt(int index) const {
    return 0.0f;
}

void BatchNormalizationLayer::saveParameters(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(bufferGamma_->contents()), bufferGamma_->length());
    os.write(reinterpret_cast<const char*>(bufferBeta_->contents()), bufferBeta_->length());
    os.write(reinterpret_cast<const char*>(bufferRunningMean_->contents()), bufferRunningMean_->length());
    os.write(reinterpret_cast<const char*>(bufferRunningVariance_->contents()), bufferRunningVariance_->length());
}

void BatchNormalizationLayer::loadParameters(std::istream& is) {
    is.read(reinterpret_cast<char*>(bufferGamma_->contents()), bufferGamma_->length());
    bufferGamma_->didModifyRange(NS::Range(0, bufferGamma_->length()));

    is.read(reinterpret_cast<char*>(bufferBeta_->contents()), bufferBeta_->length());
    bufferBeta_->didModifyRange(NS::Range(0, bufferBeta_->length()));

    is.read(reinterpret_cast<char*>(bufferRunningMean_->contents()), bufferRunningMean_->length());
    bufferRunningMean_->didModifyRange(NS::Range(0, bufferRunningMean_->length()));

    is.read(reinterpret_cast<char*>(bufferRunningVariance_->contents()), bufferRunningVariance_->length());
    bufferRunningVariance_->didModifyRange(NS::Range(0, bufferRunningVariance_->length()));
}
