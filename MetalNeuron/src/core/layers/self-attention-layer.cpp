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
#include "weight-initializer.h"

SelfAttentionLayer::SelfAttentionLayer(uint inputDim, uint modelDim, uint seqLength)
    : inputDim_(inputDim),
      modelDim_(modelDim),
      seqLength_(seqLength),
      batchSize_(0),
      device_(nullptr),
      bufferQ_(nullptr),
      bufferK_(nullptr),
      bufferV_(nullptr),
      weightsQ_(nullptr),
      weightsK_(nullptr),
      weightsV_(nullptr),
      outputProjection_(nullptr),
      optimizerWeightsQ_(nullptr),
      optimizerWeightsK_(nullptr),
      optimizerWeightsV_(nullptr),
      optimizerOutputProjection_(nullptr),
      forwardPipelineState_(nullptr),
      backwardPipelineState_(nullptr) {
}

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
    
    float* q = static_cast<float*>(weightsQ_->contents());
    float* k = static_cast<float*>(weightsK_->contents());
    float* v = static_cast<float*>(weightsV_->contents());
    float* o = static_cast<float*>(outputProjection_->contents());
    if (initializer_ == "he") {
        WeightInitializer::initializeHe(q, inputDim_, modelDim_);
        WeightInitializer::initializeHe(k, inputDim_, modelDim_);
        WeightInitializer::initializeHe(v, inputDim_, modelDim_);
        WeightInitializer::initializeHe(o, inputDim_, modelDim_);
    } else {
        WeightInitializer::initializeXavier(q, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(k, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(v, inputDim_, modelDim_);
        WeightInitializer::initializeXavier(o, inputDim_, modelDim_);
    }
    weightsQ_->didModifyRange(NS::Range(0, weightsQ_->length()));
    weightsK_->didModifyRange(NS::Range(0, weightsK_->length()));
    weightsV_->didModifyRange(NS::Range(0, weightsV_->length()));
    outputProjection_->didModifyRange(NS::Range(0, outputProjection_->length()));


    outputBuffers_[BufferType::Output] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    outputBuffers_[BufferType::OutputErrors] = device->newBuffer(activationBufferSize, MTL::ResourceStorageModeManaged);
    
    optimizerWeightsQ_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsK_->buildBuffers(device, weightsBufferSize);
    optimizerWeightsV_->buildBuffers(device, weightsBufferSize);
    optimizerOutputProjection_->buildBuffers(device, weightsBufferSize);
}

void SelfAttentionLayer::forward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    uint bs = (uint)batchSize;

    // Binding input and weight buffers
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);
    encoder->setBuffer(weightsQ_, 0, 1);
    encoder->setBuffer(weightsK_, 0, 2);
    encoder->setBuffer(weightsV_, 0, 3);
    encoder->setBuffer(outputProjection_, 0, 4);

    // Binding buffers for intermediate Q, K, V projections
    encoder->setBuffer(bufferQ_, 0, 5);
    encoder->setBuffer(bufferK_, 0, 6);
    encoder->setBuffer(bufferV_, 0, 7);

    // Binding the final output buffer
    encoder->setBuffer(outputBuffers_[BufferType::Output], 0, 8);

    // Constant parameters (dimensions)
    encoder->setBytes(&bs, sizeof(uint), 9);
    encoder->setBytes(&seqLength_, sizeof(uint), 10);
    encoder->setBytes(&inputDim_, sizeof(uint), 11);
    encoder->setBytes(&modelDim_, sizeof(uint), 12);

    // Thread dispatch configuration
    const int gridSize = batchSize * seqLength_ * modelDim_;
    MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    MTL::Size threadsPerGroup = MTL::Size(std::min(gridSize, 512), 1, 1);

    encoder->dispatchThreads(threadsPerGrid, threadsPerGroup);
    encoder->endEncoding();
}



void SelfAttentionLayer::backward(MTL::CommandBuffer* commandBuffer, int batchSize) {
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    uint bs = (uint)batchSize;

    // Binding buffers (must match exactly kernel buffer indices)
    encoder->setBuffer(inputBuffers_[BufferType::Input], 0, 0);                  // Inputs from forward pass
    encoder->setBuffer(weightsQ_, 0, 1);                                         // Q weights
    encoder->setBuffer(weightsK_, 0, 2);                                         // K weights
    encoder->setBuffer(weightsV_, 0, 3);                                         // V weights
    encoder->setBuffer(outputProjection_, 0, 4);                                 // Output projection weights
    
    encoder->setBuffer(bufferQ_, 0, 5);
    encoder->setBuffer(bufferK_, 0, 6);
    encoder->setBuffer(bufferV_, 0, 7);

    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors], 0, 8);          // Errors leaving the layer
    encoder->setBuffer(inputBuffers_[BufferType::InputErrors], 0, 9);            // Errors entering the layer


    encoder->setBuffer(optimizerWeightsQ_->gradientBuffer(), 0, 10);              // Gradients for weightsQ
    encoder->setBuffer(optimizerWeightsK_->gradientBuffer(), 0, 11);              // Gradients for weightsK
    encoder->setBuffer(optimizerWeightsV_->gradientBuffer(), 0, 12);              // Gradients for weightsV
    encoder->setBuffer(optimizerOutputProjection_->gradientBuffer(), 0, 13);     // Gradients for outputProjection

    // Constant arguments (dimensions)
    encoder->setBytes(&bs, sizeof(uint), 14);
    encoder->setBytes(&seqLength_, sizeof(uint), 15);
    encoder->setBytes(&inputDim_, sizeof(uint), 16);
    encoder->setBytes(&modelDim_, sizeof(uint), 17);

    // Thread dispatch configuration: one thread per element
    const int gridSize = batchSize * seqLength_ * inputDim_;
    MTL::Size threadsPerGrid = MTL::Size(gridSize, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(std::min(gridSize, 512), 1, 1);
    encoder->dispatchThreads(threadsPerGrid, threadgroupSize);
    
    optimizerWeightsQ_->encode(encoder, bufferQ_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsK_->encode(encoder, bufferK_, inputDim_ * modelDim_, batchSize);
    optimizerWeightsV_->encode(encoder, bufferV_, inputDim_ * modelDim_, batchSize);
    optimizerOutputProjection_->encode(encoder, outputProjection_, inputDim_ * modelDim_, batchSize);
    
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
void SelfAttentionLayer::onForwardComplete(MTL::CommandQueue*, int) {
    //Logger::log.printFloatBuffer(inputBuffers_[BufferType::InputErrors], "[Self-Attention Input Errors]", 10);
    //Logger::log.printFloatBuffer(inputBuffers_[BufferType::Input], "[Forward Self-Attention Input (full)]", seqLength_ * inputDim_);
    //Logger::log.printFloatBuffer(outputBuffers_[BufferType::Output], "[Forward Self-Attention Output]", 2);
    
}

void SelfAttentionLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    /*
    Logger::log << "input errors @" << inputBuffers_[BufferType::InputErrors] << std::endl;
    Logger::log.printFloatBuffer(inputBuffers_[BufferType::InputErrors], "[B Self-Attention Input Errors]", 10);
    Logger::log.printFloatBuffer(outputBuffers_[BufferType::OutputErrors], "[B Self-Attention Output Errors]", 10);
    
    Logger::log.printFloatBuffer(optimizerWeightsQ_->gradientBuffer(), "[Gradient Weights Q]", 10);
    Logger::log.printFloatBuffer(optimizerWeightsK_->gradientBuffer(), "[Gradient Weights K]", 10);
    Logger::log.printFloatBuffer(optimizerWeightsV_->gradientBuffer(), "[Gradient Weights V]", 10);
    Logger::log.printFloatBuffer(optimizerOutputProjection_->gradientBuffer(), "[Gradient Output Projection]", 10);
     */
}

void SelfAttentionLayer::saveParameters(std::ostream&) const {}
void SelfAttentionLayer::loadParameters(std::istream&) {}

void SelfAttentionLayer::setIsTerminal(bool isTerminal) {
    isTerminal_ = isTerminal;
}
