//
//  residual-connection-layer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#include "self-attention-layer.h"
#include "logger.h"
#include "configuration-manager.h"
#include "model-config.h"
#include "adam-optimizer.h"

SelfAttentionLayer::SelfAttentionLayer(MTL::Device* device, int inputDim, int seqLength, int modelDim)
    : device_(device), inputDim_(inputDim), seqLength_(seqLength), modelDim_(modelDim),
      isTerminal_(false),
      bufferQ_(nullptr), bufferK_(nullptr), bufferV_(nullptr),
      weightsQ_(nullptr), weightsK_(nullptr), weightsV_(nullptr),
      outputProjection_(nullptr),
      forwardPipelineState_(nullptr), backwardPipelineState_(nullptr),
      optimizerWeightsQ_(nullptr), optimizerWeightsK_(nullptr),
      optimizerWeightsV_(nullptr), optimizerOutputProjection_(nullptr) {}

SelfAttentionLayer::~SelfAttentionLayer() {
    if (bufferQ_) bufferQ_->release();
    if (bufferK_) bufferK_->release();
    if (bufferV_) bufferV_->release();

    if (weightsQ_) weightsQ_->release();
    if (weightsK_) weightsK_->release();
    if (weightsV_) weightsV_->release();
    if (outputProjection_) outputProjection_->release();

    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void SelfAttentionLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;

    auto kernelFn = library->newFunction(NS::String::string("forward_self_attention", NS::ASCIIStringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(kernelFn, &error);
    kernelFn->release();

    if (!forwardPipelineState_) {
        Logger::log << "Error occurred creating forward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        assert(false);
    }

    auto kernelBackwardFn = library->newFunction(NS::String::string("backward_self_attention", NS::ASCIIStringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(kernelBackwardFn, &error);
    kernelBackwardFn->release();

    if (!backwardPipelineState_) {
        Logger::log << "Error occurred creating self_attention_backward pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        std::exit(-1);
    }
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto parameters = pConfig->training.optimizer.parameters;

    float lr      = pConfig->training.optimizer.learning_rate;
    float beta1   = parameters["beta1"].get_value_or<float>(0.9f);
    float beta2   = parameters["beta2"].get_value_or<float>(0.999f);
    float epsilon = parameters["epsilon"].get_value_or<float>(1e-8);
        
    optimizerWeightsQ_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerWeightsK_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerWeightsV_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerOutputProjection_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    
    optimizerWeightsQ_->buildPipeline(device, library);
    optimizerWeightsK_->buildPipeline(device, library);
    optimizerWeightsV_->buildPipeline(device, library);
    optimizerOutputProjection_->buildPipeline(device, library);
}

void SelfAttentionLayer::buildBuffers(MTL::Device* device) {
    const size_t activationBufferSize = seqLength_ * modelDim_ * sizeof(float);
    const size_t weightsBufferSize = inputDim_ * modelDim_ * sizeof(float);

    // Buffers for projected queries (Q), keys (K), and values (V)
    bufferQ_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    bufferK_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    bufferV_ = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);

    // Buffers for weights of Q, K, V projections and the output projection
    weightsQ_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    weightsK_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    weightsV_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);
    outputProjection_ = device->newBuffer(weightsBufferSize, MTL::ResourceStorageModeManaged);

    // Input and output buffers (no timestep logic)
    inputBuffers_[BufferType::Input] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    inputBuffers_[BufferType::InputErrors] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);

    outputBuffers_[BufferType::Output] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutputErrors] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
}


void SelfAttentionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);

    // Binding buffers
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(weightsQ_, 0, 1);
    encoder->setBuffer(weightsK_, 0, 2);
    encoder->setBuffer(weightsV_, 0, 3);
    encoder->setBuffer(outputProjection_, 0, 4);
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 5);
    encoder->setBuffer(bufferQ_, 0, 6);
    encoder->setBuffer(bufferK_, 0, 7);
    encoder->setBuffer(bufferV_, 0, 8);

    // Constant arguments (dimensions)
    encoder->setBytes(&batchSize, sizeof(int), 9);
    encoder->setBytes(&seqLength_, sizeof(int), 10);
    encoder->setBytes(&inputDim_, sizeof(int), 11);
    encoder->setBytes(&modelDim_, sizeof(int), 12);

    const int gridSize = batchSize * seqLength_ * modelDim_;
    const MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    const MTL::Size threadsPerGroup = MTL::Size(std::min(gridSize, 512), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadsPerGroup);
    encoder->endEncoding();
}

void SelfAttentionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);

    // Binding buffers
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors], 0, 0);
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 1);
    encoder->setBuffer(weightsQ_, 0, 2);
    encoder->setBuffer(weightsK_, 0, 3);
    encoder->setBuffer(weightsV_, 0, 4);
    encoder->setBuffer(outputProjection_, 0, 5);
    encoder->setBuffer(inputBuffers_[BufferType::InputErrors], 0, 6);
    encoder->setBuffer(optimizerWeightsQ_->gradientBuffer(), 0, 7);
    encoder->setBuffer(optimizerWeightsK_->gradientBuffer(), 0, 8);
    encoder->setBuffer(optimizerWeightsV_->gradientBuffer(), 0, 9);
    encoder->setBuffer(optimizerOutputProjection_->gradientBuffer(), 0, 10);

    // Constant arguments (dimensions)
    encoder->setBytes(&batchSize, sizeof(int), 11);
    encoder->setBytes(&seqLength_, sizeof(int), 12);
    encoder->setBytes(&inputDim_, sizeof(int), 13);
    encoder->setBytes(&modelDim_, sizeof(int), 14);

    const int gridSize = batchSize * seqLength_ * inputDim_;
    const MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    const MTL::Size threadsPerGroup = MTL::Size(std::min(gridSize, 512), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadsPerGroup);
    encoder->endEncoding();
}

void SelfAttentionLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    inputBuffers_[type] = buffer;
}

MTL::Buffer* SelfAttentionLayer::getOutputBufferAt(BufferType type, int) { return outputBuffers_[type]; }
void SelfAttentionLayer::setOutputBufferAt(BufferType type, int, MTL::Buffer* buffer) {
    outputBuffers_[type] = buffer;
}
MTL::Buffer* SelfAttentionLayer::getInputBufferAt(BufferType type, int) { return inputBuffers_[type]; }


void SelfAttentionLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

void SelfAttentionLayer::connectBackwardConnections(Layer* prevLayer,
                                   Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    if (prevLayer) {
        prevLayer->setInputBufferAt(BufferType::InputErrors, timestep, getOutputBufferAt(BufferType::OutputErrors, timestep));
    }
}

void SelfAttentionLayer::debugLog() {}
void SelfAttentionLayer::onForwardComplete(MTL::CommandQueue*, int) {}
void SelfAttentionLayer::onBackwardComplete(MTL::CommandQueue*, int) {}

void SelfAttentionLayer::saveParameters(std::ostream&) const {}
void SelfAttentionLayer::loadParameters(std::istream&) {}

void SelfAttentionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
