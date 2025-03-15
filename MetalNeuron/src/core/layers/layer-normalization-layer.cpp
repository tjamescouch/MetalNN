//
//  batch-normalization-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "input-layer.h"
#include "adam-optimizer.h"
#include "layer-normalization-layer.h"
#include <cassert>
#include <random>
#include <cstring>
#include <iostream>
#include "training-manager.h"
#include "math-lib.h"
#include "logger.h"
#include "model-config.h"
#include "configuration-manager.h"

LayerNormalizationLayer::LayerNormalizationLayer(int featureDim, int batchSize, float learningRate, float epsilon)
    : featureDim_(featureDim),
      batchSize_(batchSize),
      learningRate_(learningRate),
      epsilon_(epsilon),
      bufferDebug_(nullptr),
      bufferGamma_(nullptr),
      bufferBeta_(nullptr),
      bufferSavedMean_(nullptr),
      bufferSavedVariance_(nullptr),
      forwardPipelineState_(nullptr),
      backwardPipelineState_(nullptr)
{
}

LayerNormalizationLayer::~LayerNormalizationLayer() {
    if (bufferDebug_) bufferDebug_->release();
    if (bufferGamma_) bufferGamma_->release();
    if (bufferBeta_) bufferBeta_->release();
    
    if (bufferSavedMean_) {
        bufferSavedMean_->release();
        bufferSavedMean_ = nullptr;
    }
    if (bufferSavedVariance_) {
        bufferSavedVariance_->release();
        bufferSavedVariance_ = nullptr;
    }
    
    for (auto ob : outputBuffers_) {
        ob.second[0]->release();
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void LayerNormalizationLayer::initializeParameters(MTL::Device* device) {
    std::vector<float> debug(featureDim_, 0.0f);
    std::vector<float> gamma(featureDim_, 1.0f);
    std::vector<float> beta(featureDim_, 0.0f);

    // This is the size for the per-feature arrays:
    size_t bufferSize = sizeof(float) * featureDim_;

    bufferDebug_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferGamma_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    bufferSavedMean_ = device->newBuffer(sizeof(float) * batchSize_, MTL::ResourceStorageModeManaged);
    bufferSavedVariance_ = device->newBuffer(sizeof(float) * batchSize_, MTL::ResourceStorageModeManaged);

    // Initialize all to zeros or ones as appropriate:
    memcpy(bufferDebug_->contents(), debug.data(), bufferSize);
    memcpy(bufferGamma_->contents(), gamma.data(), bufferSize);
    memcpy(bufferBeta_->contents(), beta.data(), bufferSize);

    // Also zero out the "saved" stats buffers so they start in a known state
    std::vector<float> zeros(featureDim_, 0.0f);
    memcpy(bufferSavedMean_->contents(), zeros.data(), bufferSize);
    memcpy(bufferSavedVariance_->contents(), zeros.data(), bufferSize);

    // Mark them as modified
    bufferDebug_->didModifyRange(NS::Range(0, bufferSize));
    bufferGamma_->didModifyRange(NS::Range(0, bufferSize));
    bufferBeta_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedMean_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedVariance_->didModifyRange(NS::Range(0, bufferSize));
}

void LayerNormalizationLayer::buildBuffers(MTL::Device* device) {
    size_t bufferSize = batchSize_ * featureDim_ * sizeof(float);

    std::vector<float> gamma(featureDim_, 1.0f); // scale initialized to 1
    std::vector<float> beta(featureDim_, 0.0f);  // shift initialized to 0

    bufferGamma_ = device->newBuffer(sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(sizeof(float) * featureDim_, MTL::ResourceStorageModeManaged);

    memcpy(bufferGamma_->contents(), gamma.data(), sizeof(float) * featureDim_);
    memcpy(bufferBeta_->contents(), beta.data(), sizeof(float) * featureDim_);

    bufferGamma_->didModifyRange(NS::Range(0, sizeof(float) * featureDim_));
    bufferBeta_->didModifyRange(NS::Range(0, sizeof(float) * featureDim_));

    // Intermediate buffers for per-sample statistics
    bufferSavedMean_ = device->newBuffer(sizeof(float) * batchSize_, MTL::ResourceStorageModeManaged);
    bufferSavedVariance_ = device->newBuffer(sizeof(float) * batchSize_, MTL::ResourceStorageModeManaged);

    // Debug buffer (optional)
    bufferDebug_ = device->newBuffer(sizeof(float) * 256, MTL::ResourceStorageModeManaged);

    // Allocate input and output buffers explicitly (single timestep only)
    inputBuffers_[BufferType::Input].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::Output].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    inputBuffers_[BufferType::InputErrors].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::OutputErrors].push_back(device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged));

    // Optimizer buffers for gamma and beta
    optimizerGamma_->buildBuffers(device, featureDim_ * sizeof(float));
    optimizerBeta_->buildBuffers(device, featureDim_ * sizeof(float));
}

void LayerNormalizationLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto forwardFn = library->newFunction(NS::String::string("forward_layer_norm", NS::UTF8StringEncoding));
    assert(forwardFn && "Forward function not found.");

    auto backwardFn = library->newFunction(NS::String::string("backward_layer_norm", NS::UTF8StringEncoding));
    assert(backwardFn && "Backward function not found.");

    forwardPipelineState_ = device->newComputePipelineState(forwardFn, &error);
    assert(forwardPipelineState_);

    backwardPipelineState_ = device->newComputePipelineState(backwardFn, &error);
    assert(backwardPipelineState_);

    forwardFn->release();
    backwardFn->release();

    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto optimizerConfig = pConfig->training.optimizer;

    optimizerGamma_ = std::make_unique<AdamOptimizer>(learningRate_, optimizerConfig.beta1, optimizerConfig.beta2, optimizerConfig.epsilon);
    optimizerBeta_  = std::make_unique<AdamOptimizer>(learningRate_, optimizerConfig.beta1, optimizerConfig.beta2, optimizerConfig.epsilon);

    optimizerGamma_->buildPipeline(device, library);
    optimizerBeta_->buildPipeline(device, library);
}

void LayerNormalizationLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);

    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);        // input
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);      // output
    encoder->setBuffer(bufferGamma_, 0, 2);
    encoder->setBuffer(bufferBeta_, 0, 3);
    encoder->setBuffer(bufferSavedMean_, 0, 4);
    encoder->setBuffer(bufferSavedVariance_, 0, 5);
    encoder->setBytes(&epsilon_, sizeof(float), 6);
    encoder->setBytes(&featureDim_, sizeof(int), 7);
    encoder->setBytes(&batchSize_, sizeof(int), 8);
    encoder->setBuffer(bufferDebug_, 0, 9);

    // Explicit thread indexing per sample rather than per feature
    MTL::Size threadsPerGroup = MTL::Size(std::min(batchSize_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((batchSize_ + 1023) / 1024, 1, 1);
    
    encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
    encoder->endEncoding();
}

void LayerNormalizationLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    // indices:
    encoder->setBuffer(inputBuffers_[BufferType::Input][0],       0, 0);
    encoder->setBuffer(inputBuffers_[BufferType::InputErrors][0], 0, 1);
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][0], 0, 2);
    encoder->setBuffer(bufferGamma_, 0, 3);
    encoder->setBuffer(bufferBeta_, 0, 4);

    // NEW: savedMean=5, savedVariance=6
    encoder->setBuffer(bufferSavedMean_, 0, 5);
    encoder->setBuffer(bufferSavedVariance_, 0, 6);

    // SHIFT the rest:
    encoder->setBytes(&epsilon_,       sizeof(float), 7);
    encoder->setBytes(&featureDim_,    sizeof(int),   8);
    encoder->setBytes(&batchSize,      sizeof(uint),  9);
    encoder->setBytes(&learningRate_,  sizeof(float), 10);
    
    encoder->setBuffer(optimizerBeta_->gradientBuffer(), 0, 11);
    encoder->setBuffer(optimizerGamma_->gradientBuffer(), 0, 12);
    
    optimizerGamma_->encode(encoder, bufferGamma_, featureDim_, batchSize);
    optimizerBeta_->encode(encoder, bufferBeta_, featureDim_, batchSize);

    MTL::Size threadsPerGroup = MTL::Size(std::min(featureDim_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((featureDim_ + 1023) / 1024, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
    encoder->endEncoding();
}


void LayerNormalizationLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
}

void LayerNormalizationLayer::updateTargetBufferAt(const float* targetData, int timestep, int batchSize) {
    assert(timestep==0 && "Timesteps not supported for this layer");
}


void LayerNormalizationLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* LayerNormalizationLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    return outputBuffers_[type][timestep];
}

void LayerNormalizationLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* LayerNormalizationLayer::getInputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    return inputBuffers_[type][timestep];
}

void LayerNormalizationLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer, MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

void LayerNormalizationLayer::connectBackwardConnections(Layer* prevLayer, Layer* inputLayer, MTL::Buffer* zeroBuffer, int timestep) {
    prevLayer->setInputBufferAt(BufferType::InputErrors, 0, getOutputBufferAt(BufferType::OutputErrors, timestep));
}

void LayerNormalizationLayer::saveParameters(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(bufferGamma_->contents()), bufferGamma_->length());
    os.write(reinterpret_cast<const char*>(bufferBeta_->contents()), bufferBeta_->length());
}

void LayerNormalizationLayer::loadParameters(std::istream& is) {
    is.read(reinterpret_cast<char*>(bufferGamma_->contents()), bufferGamma_->length());
    bufferGamma_->didModifyRange(NS::Range(0, bufferGamma_->length()));

    is.read(reinterpret_cast<char*>(bufferBeta_->contents()), bufferBeta_->length());
    bufferBeta_->didModifyRange(NS::Range(0, bufferBeta_->length()));
}

void LayerNormalizationLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
}

void LayerNormalizationLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {

}
