#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "adam-optimizer.h"
#include "dense-layer.h"
#include "common.h"

DenseLayer::DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activation)
: inputDim_(inputDim), outputDim_(outputDim), sequenceLength_(1), activation_(activation),
bufferWeights_(nullptr), bufferBias_(nullptr), bufferDecay_(nullptr),isTerminal_(false),
forwardPipelineState_(nullptr), backwardPipelineState_(nullptr)
{
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_, nullptr);
}

DenseLayer::~DenseLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ib : inputBuffers_) {
            ib.second[t]->release();
        }
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (bufferWeights_) bufferWeights_->release();
    if (bufferBias_) bufferBias_->release();
    if (bufferDecay_) bufferDecay_->release();
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void DenseLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    auto forwardFunc = library->newFunction(NS::String::string("forward_dense_layer", NS::UTF8StringEncoding));
    assert(forwardFunc && "Forward function not found.");
    
    auto backwardFunc = library->newFunction(NS::String::string("learn_dense_layer", NS::UTF8StringEncoding));
    assert(backwardFunc && "Backward function not found.");
    
    NS::Error* error = nullptr;
    forwardPipelineState_ = device->newComputePipelineState(forwardFunc, &error);
    assert(forwardPipelineState_);
    
    backwardPipelineState_ = device->newComputePipelineState(backwardFunc, &error);
    assert(backwardPipelineState_);
    
    forwardFunc->release();
    backwardFunc->release();
    
    
    optimizerWeights_ = std::make_unique<AdamOptimizer>(0.001f, 0.9f, 0.999f, 1e-8f);
    optimizerBiases_ = std::make_unique<AdamOptimizer>(0.001f, 0.9f, 0.999f, 1e-8f);
    
    optimizerWeights_->buildPipeline(device, library);
    optimizerBiases_->buildPipeline(device, library);
}

#include "weight-initializer.h"

void DenseLayer::buildBuffers(MTL::Device* device) {
    const float decay = 1.0f;
    
    size_t weightSize = inputDim_ * outputDim_ * sizeof(float);
    size_t biasSize = outputDim_ * sizeof(float);
    
    bufferWeights_ = device->newBuffer(weightSize, MTL::ResourceStorageModeManaged);
    float* w = static_cast<float*>(bufferWeights_->contents());
    WeightInitializer::initializeXavier(w, inputDim_, outputDim_);
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    bufferBias_ = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    WeightInitializer::initializeBias(b, outputDim_);
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    
    
    bufferDecay_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferDecay_->contents(), &decay, sizeof(float));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
    
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Delta].resize(sequenceLength_);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::Debug].resize(sequenceLength_);
    
    
    int t = 0;
    
    outputBuffers_[BufferType::Delta][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Delta][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Delta][t]->length()));
    
    
    outputBuffers_[BufferType::OutputErrors][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::Output][t] = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    outputBuffers_[BufferType::Debug][t]   = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    inputBuffers_[BufferType::Targets][t]  = device->newBuffer(outputDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    
    memset(outputBuffers_[BufferType::Output][t]->contents(), 0, outputDim_ * sizeof(float));
    memset(inputBuffers_[BufferType::Targets][t]->contents(), 0, outputDim_ * sizeof(float));
    memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, outputDim_ * sizeof(float));
    memset(outputBuffers_[BufferType::Debug][t]->contents(), 0, outputDim_ * sizeof(float));
    
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
    memcpy(inputBuffers_[BufferType::Targets][timestep]->contents(), targetData, outputDim_ * sizeof(float));
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, outputDim_ * sizeof(float)));
}

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);
    
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
        
        MTL::Size threadsPerThreadgroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
        MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
        encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        encoder->endEncoding();
    }
}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf) {
    int t = 0;
    memset(outputBuffers_[BufferType::Delta][t]->contents(), 0, outputBuffers_[BufferType::Delta][t]->length());
    outputBuffers_[BufferType::Delta][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::Delta][t]->length()));
    
    uint activationRaw = static_cast<uint>(activation_);
    
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][t], 0, 0);         // Input activations (h)
    encoder->setBuffer(bufferWeights_, 0, 1);           // Weights (W)
    encoder->setBuffer(bufferBias_, 0, 2);              // Biases (b)
    encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 3);        // Target (y)
    encoder->setBuffer(inputBuffers_[BufferType::Targets][t], 0, 4);        // Predicted output (y_hat)
    encoder->setBuffer(outputBuffers_[BufferType::Delta][t], 0, 5);         // Error (delta) ??
    encoder->setBytes(&inputDim_, sizeof(uint), 6);     // Input dimension (pH)
    encoder->setBytes(&outputDim_, sizeof(uint), 7);    // Output dimension (pN)
    encoder->setBuffer(bufferDecay_, 0, 8);             // Decay factor (pDecay)
    encoder->setBytes(&activationRaw, sizeof(uint), 9); // Activation function type
    encoder->setBuffer(outputBuffers_[BufferType::Debug][t], 0, 10);  // debug signal
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 11);
    
    MTL::Size threadsPerThreadgroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
    
    encoder->endEncoding();
    
    // Sync buffer contents back to CPU (existing logic preserved)
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
    
    for (int t = 0; t < sequenceLength_; ++t) {
        inputBuffers_[BufferType::InputErrors][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::InputErrors][t]->length()));
    }
}

void DenseLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* DenseLayer::getInputBufferAt(BufferType type, int timestep) {
    auto it = inputBuffers_.find(type);
    if (it != inputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;  // Explicitly handle missing keys appropriately
}

MTL::Buffer* DenseLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(timestep == 0);
    auto it = outputBuffers_.find(type);
    if (it != outputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr;  // Explicitly handle the error scenario as needed
}

int DenseLayer::outputSize() const {
    return outputDim_;
}


void DenseLayer::connectInputBuffers(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}


int DenseLayer::getParameterCount() const {
    return inputDim_ * outputDim_ + outputDim_;  // weights + biases
}

float DenseLayer::getParameterAt(int index) const {
    float* weights = static_cast<float*>(bufferWeights_->contents());
    float* biases = static_cast<float*>(bufferBias_->contents());
    
    if (index < inputDim_ * outputDim_) {
        return weights[index];
    } else {
        return biases[index - inputDim_ * outputDim_];
    }
}

void DenseLayer::setParameterAt(int index, float value) {
    float* weights = static_cast<float*>(bufferWeights_->contents());
    float* biases = static_cast<float*>(bufferBias_->contents());
    
    if (index < inputDim_ * outputDim_) {
        weights[index] = value;
        bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    } else {
        biases[index - inputDim_ * outputDim_] = value;
        bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    }
}

float DenseLayer::getGradientAt(int index) const {
    float* deltaBuffer = static_cast<float*>(outputBuffers_.at(BufferType::Delta)[0]->contents());
    
    // You will need to have stored gradients explicitly in buffers.
    return deltaBuffer[index];  // Simplified for illustration; adjust accordingly
}

void DenseLayer::saveParameters(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(bufferWeights_->contents()), bufferWeights_->length());
    os.write(reinterpret_cast<const char*>(bufferBias_->contents()), bufferBias_->length());
    os.write(reinterpret_cast<const char*>(bufferDecay_->contents()), bufferDecay_->length());
}

void DenseLayer::loadParameters(std::istream& is) {
    is.read(reinterpret_cast<char*>(bufferWeights_->contents()), bufferWeights_->length());
    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    
    is.read(reinterpret_cast<char*>(bufferBias_->contents()), bufferBias_->length());
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    is.read(reinterpret_cast<char*>(bufferDecay_->contents()), bufferDecay_->length());
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
}

void DenseLayer::debugLog() {
#ifdef DEBUG_DENSE_LAYER
    float* inputs = static_cast<float*>(inputBuffers_[BufferType::Input][0]->contents());
    printf("[DenseLayer Input Debug] timestep %d: ", 0);
    for(int i = 0; i < inputBuffers_[BufferType::Input][0]->length()/sizeof(float); ++i)
        printf(" %f, ", inputs[i]);
    printf("\n");
    
    float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][0]->contents());
    printf("[DenseLayer Output Debug] timestep %d: ", 0);
    for(int i = 0; i < outputBuffers_[BufferType::Output][0]->length()/sizeof(float); ++i)
        printf(" %f, ", outputs[i]);
    printf("\n");
    
    float* weights = static_cast<float*>(bufferWeights_->contents());
    printf("[DenseLayer DebugLog] Weights sample: %f, %f, %f\n", weights[0], weights[1], weights[2]);
    
    // Optionally log biases or other important states:
    float* biases = static_cast<float*>(bufferBias_->contents());
    printf("[DenseLayer DebugLog] Biases sample: %f, %f, %f\n", biases[0], biases[1], biases[2]);
    
    float* decay = static_cast<float*>(bufferDecay_->contents());
    printf("[DenseLayer DebugLog] Decay factor: %f\n", *decay);
    
#endif
}

void DenseLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue) {
    auto cmdBuf = _pCommandQueue->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    optimizerWeights_->encode(encoder, bufferWeights_, inputDim_ * outputDim_);
    optimizerBiases_->encode(encoder, bufferBias_, outputDim_);

    encoder->endEncoding();
}
