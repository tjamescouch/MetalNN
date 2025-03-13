#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "weight-initializer.h"
#include "configuration-manager.h"
#include "math-lib.h"
#include "adam-optimizer.h"
#include "dense-layer.h"
#include "common.h"
#include "logger.h"

DenseLayer::DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activation, int batchSize)
: inputDim_(inputDim), outputDim_(outputDim), sequenceLength_(1), activation_(activation),
bufferWeights_(nullptr), bufferBias_(nullptr), bufferDecay_(nullptr), bufferLearningRate_(nullptr), isTerminal_(false),
forwardPipelineState_(nullptr), backwardPipelineState_(nullptr), learningRate_(0.001), batchSize_(batchSize)
{
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_, nullptr);
    
    layerIndex = ++layerCounter;
}

DenseLayer::~DenseLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (bufferWeights_) bufferWeights_->release();
    if (bufferBias_) bufferBias_->release();
    if (bufferDecay_) bufferDecay_->release();
    if (bufferLearningRate_) bufferLearningRate_->release();
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_dense_layer", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    auto backwardFunc = library->newFunction(NS::String::string(isTerminal_ ? "learn_terminal_dense_layer" : "learn_non_terminal_dense_layer", NS::UTF8StringEncoding));
    assert(backwardFunc && "Backward function not found.");
    
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);
    
    forwardFunc->release();
    backwardFunc->release();
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto parameters = pConfig->training.optimizer.parameters;

    float lr      = pConfig->training.optimizer.learning_rate;
    float beta1   = parameters["beta1"].get_value_or<float>(0.9f);
    float beta2   = parameters["beta2"].get_value_or<float>(0.999f);
    float epsilon = parameters["epsilon"].get_value_or<float>(1e-8);
        
    optimizerWeights_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerBiases_  = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    
    optimizerWeights_->buildPipeline(device, library);
    optimizerBiases_->buildPipeline(device, library);
}


void DenseLayer::buildBuffers(MTL::Device* device) {
    const float decay = 1.0f;
    
    size_t weightSize = inputDim_ * outputDim_ * sizeof(float);
    size_t biasSize = outputDim_ * sizeof(float);
    
    bufferWeights_ = device->newBuffer(weightSize, MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    if (initializer_ == "he") {
        WeightInitializer::initializeHe(w, inputDim_, outputDim_);
    } else {
        WeightInitializer::initializeXavier(w, inputDim_, outputDim_);
    }
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    WeightInitializer::initializeBias(b, outputDim_);
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    bufferDecay_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferDecay_->contents(), &decay, sizeof(float));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
    
    bufferLearningRate_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferLearningRate_->contents(), &learningRate_, sizeof(float));
    bufferLearningRate_->didModifyRange(NS::Range(0, bufferLearningRate_->length()));
    
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Delta].resize(sequenceLength_);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    
    int t = 0;

    outputBuffers_[BufferType::Delta][t] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Delta][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Delta][t]->length()));

    outputBuffers_[BufferType::OutputErrors][t] = device->newBuffer(inputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Output][t] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);

    outputBuffers_[BufferType::Debug][t] = device->newBuffer(inputDim_ * outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    inputBuffers_[BufferType::Targets][t] = device->newBuffer(outputDim_ * batchSize_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    memset(outputBuffers_[BufferType::Output][t]->contents(), 0, outputDim_ * batchSize_ * sizeof(float));
    memset(inputBuffers_[BufferType::Targets][t]->contents(), 0, outputDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, inputDim_ * batchSize_ * sizeof(float));
    memset(outputBuffers_[BufferType::Debug][t]->contents(), 0, inputDim_ * outputDim_ * sizeof(float));
    
    inputBuffers_[BufferType::Targets][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][t]->length()));
    outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Output][t]->length()));
    outputBuffers_[BufferType::OutputErrors][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutputErrors][t]->length()));
    outputBuffers_[BufferType::Debug][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Debug][t]->length()));

    optimizerWeights_->buildBuffers(device, weightSize);
    optimizerBiases_->buildBuffers(device, biasSize);
}

void DenseLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep == 0);
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

void DenseLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(timestep == 0);
    memcpy(inputBuffers_[BufferType::Targets][timestep]->contents(), targetData, inputBuffers_[BufferType::Targets][timestep]->length());
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][timestep]->length()));
}

void DenseLayer::updateTargetBufferAt(const float* targetData, int timestep, int _batchSize) {
    assert(timestep == 0);
    memcpy(inputBuffers_[BufferType::Targets][timestep]->contents(), targetData, batchSize_ * outputDim_ * sizeof(float));
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Targets][timestep]->length()));
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf, int _batchSize) {
    uint activationRaw = static_cast<uint>(activation_);
    uint bs = (uint)batchSize_;
        
    for (int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 1);
        encoder->setBuffer(bufferWeights_, 0, 2);
        encoder->setBuffer(bufferBias_, 0, 3);
        encoder->setBytes(&inputDim_, sizeof(int), 4);
        encoder->setBytes(&outputDim_, sizeof(int), 5);
        encoder->setBytes(&activationRaw, sizeof(uint), 6);
        encoder->setBytes(&bs, sizeof(uint), 7);
        encoder->setBuffer(outputBuffers_[BufferType::Debug][0], 0, 8);
                
        uint gridSize = batchSize_ * outputDim_;
        uint threadsPerThreadgroup = std::min<uint>(1024, gridSize);

        MTL::Size threadgroupSize(threadsPerThreadgroup, 1, 1);
        MTL::Size threadgroups((gridSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
        encoder->endEncoding();
        
        inputBuffers_[BufferType::Input][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][0]->length()));
    }
}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf, int _batchSize) {    
    uint activationRaw = static_cast<uint>(activation_);
    uint bs = (uint)batchSize_;

    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    // Binding buffers
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);
    encoder->setBuffer(bufferWeights_, 0, 1);
    encoder->setBuffer(bufferBias_, 0, 2);
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 3);
    encoder->setBuffer(isTerminal_ ? inputBuffers_[BufferType::Targets][0] : inputBuffers_[BufferType::InputErrors][0], 0, 4);
    encoder->setBuffer(outputBuffers_[BufferType::Delta][0], 0, 5);
    encoder->setBytes(&inputDim_, sizeof(uint), 6);
    encoder->setBytes(&outputDim_, sizeof(uint), 7);
    encoder->setBuffer(bufferDecay_, 0, 8);
    encoder->setBytes(&activationRaw, sizeof(uint), 9);
    encoder->setBuffer(outputBuffers_[BufferType::Debug][0], 0, 10);
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][0], 0, 11);
    encoder->setBytes(&bs, sizeof(uint), 12);
    encoder->setBuffer(bufferLearningRate_, 0, 13);

    uint gridSize = batchSize_ * outputDim_;
    uint threadsPerThreadgroup = std::min<uint>(1024, gridSize);

    MTL::Size threadgroupSize(threadsPerThreadgroup, 1, 1);
    MTL::Size threadgroups((gridSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
    encoder->endEncoding();

    inputBuffers_[BufferType::Input][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::Input][0]->length()));
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));

}

void DenseLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* DenseLayer::getInputBufferAt(BufferType type, int timestep) {
    return inputBuffers_[type][timestep];
}

MTL::Buffer* DenseLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(timestep == 0);
    return outputBuffers_[type][timestep];
}

int DenseLayer::inputSize() const {
    return inputDim_;
}

int DenseLayer::outputSize() const {
    return outputDim_;
}

void DenseLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

void DenseLayer::connectBackwardConnections(Layer* prevLayer,
                                   Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    if (prevLayer) {
        prevLayer->setInputBufferAt(BufferType::InputErrors, timestep, getOutputBufferAt(BufferType::OutputErrors, timestep));
    }
}

void DenseLayer::saveParameters(std::ostream& os) const { //FIXME encode buffer lengths
    os.write(reinterpret_cast<const char*>(bufferWeights_->contents()), bufferWeights_->length());
    os.write(reinterpret_cast<const char*>(bufferBias_->contents()), bufferBias_->length());
}

void DenseLayer::loadParameters(std::istream& is) { //FIXME - decode buffer lengths and verify
    is.read(reinterpret_cast<char*>(bufferWeights_->contents()), bufferWeights_->length());
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    is.read(reinterpret_cast<char*>(bufferBias_->contents()), bufferBias_->length());
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
}

void DenseLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    inputBuffers_[BufferType::InputErrors][0]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::InputErrors][0]->length()));
}


void DenseLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    
    auto cmdBuf = _pCommandQueue->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    optimizerWeights_->encode(encoder, bufferWeights_, inputDim_ * outputDim_, batchSize);
    optimizerBiases_->encode(encoder, bufferBias_, outputDim_, batchSize);

    encoder->endEncoding();
    
    memset(outputBuffers_[BufferType::OutputErrors][0]->contents(), 0, outputBuffers_[BufferType::OutputErrors][0]->length());
    outputBuffers_[BufferType::OutputErrors][0]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutputErrors][0]->length()));
}

void DenseLayer::debugLog() {

}

int DenseLayer::layerCounter = 0;
