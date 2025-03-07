#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "configuration-manager.h"
#include "math-lib.h"
#include "adam-optimizer.h"
#include "dense-layer.h"
#include "common.h"

DenseLayer::DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activation)
: inputDim_(inputDim), outputDim_(outputDim), sequenceLength_(1), activation_(activation),
bufferWeights_(nullptr), bufferBias_(nullptr), bufferDecay_(nullptr), bufferLearningRate_(nullptr), isTerminal_(false),
forwardPipelineState_(nullptr), backwardPipelineState_(nullptr), learningRate_(0.001)
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

void DenseLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize) {
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
        
        // Define threadgroup size respecting the maximum (1024)
        const uint maxThreadsPerGroup = 1024;
        MTL::Size threadgroupSize = MTL::Size(mathlib::min<int>(inputDim_, maxThreadsPerGroup), 1, 1);

        // Compute the required number of threadgroups
        MTL::Size gridSize = MTL::Size(inputDim_, 1, 1);
        MTL::Size threadgroups = MTL::Size((inputDim_ + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

        // Dispatch threads correctly
        encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
        encoder->endEncoding();
    }
}

void DenseLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    uint activationRaw = static_cast<uint>(activation_);

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
    encoder->setBytes(&batchSize, sizeof(uint), 12);
    encoder->setBuffer(bufferLearningRate_, 0, 13);

    // Corrected Dispatch Logic
    const uint gridSize = outputDim_ * batchSize;
    const uint maxThreadsPerGroup = 1024;
    const uint threadsPerThreadgroup = std::min(gridSize, maxThreadsPerGroup);

    MTL::Size threadgroupSize = MTL::Size(threadsPerThreadgroup, 1, 1);
    MTL::Size threadgroups = MTL::Size((gridSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1);

    encoder->dispatchThreadgroups(threadgroups, threadgroupSize);
    encoder->endEncoding();

    bufferWeights_->didModifyRange(NS::Range(0, bufferWeights_->length()));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));

    for (int t = 0; t < sequenceLength_; ++t) {
        inputBuffers_[BufferType::InputErrors][t]->didModifyRange(NS::Range(0, inputBuffers_[BufferType::InputErrors][t]->length()));
        outputBuffers_[BufferType::OutputErrors][t]->didModifyRange(NS::Range(0, outputBuffers_[BufferType::OutputErrors][t]->length()));
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
    prevLayer->setInputBufferAt(BufferType::InputErrors, 0, getOutputBufferAt(BufferType::OutputErrors, timestep));
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
    
    float* biases = static_cast<float*>(bufferBias_->contents());
    printf("[DenseLayer DebugLog] Biases sample: %f, %f, %f\n", biases[0], biases[1], biases[2]);
    
    float* decay = static_cast<float*>(bufferDecay_->contents());
    printf("[DenseLayer DebugLog] Decay factor: %f\n", *decay);
#endif
#ifdef DEBUG_OUTPUT_ERRORS
    {
        float* outputErrors = static_cast<float*>(outputBuffers_[BufferType::OutputErrors][0]->contents());
        size_t outputErrorCount = outputBuffers_[BufferType::OutputErrors][0]->length() / sizeof(float);
        
        for (int i = 0; i < outputErrorCount; ++i) {
            float grad_value = ((float*)outputBuffers_[BufferType::OutputErrors][0]->contents())[i];
            printf("outputBuffers_[BufferType::OutputErrors][0]->contents())[%d]: %f\n", i, grad_value);
        }
    }
#endif
#ifdef DEBUG_L2_NORMS
    {
        float* outputErrors = static_cast<float*>(outputBuffers_[BufferType::OutputErrors][0]->contents());
        size_t outputErrorCount = outputBuffers_[BufferType::OutputErrors][0]->length() / sizeof(float);
        
        float outputErrorNorm = 0.0f;
        for (size_t i = 0; i < outputErrorCount; ++i)
            outputErrorNorm += outputErrors[i] * outputErrors[i];
        
        outputErrorNorm = sqrtf(outputErrorNorm);
        this->isTerminal_ ? printf("[Terminal DenseLayer DebugLog] Output Error Gradient L2 Norm: %f\n", outputErrorNorm) : printf("[DenseLayer DebugLog] Output Error Gradient L2 Norm: %f\n", outputErrorNorm);
    }
#endif
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
