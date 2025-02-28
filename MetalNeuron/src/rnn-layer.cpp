// rnn-layer.cpp (BPTT updates)
#include <iostream>
#include "rnn-layer.h"
#include "common.h"

RNNLayer::RNNLayer(int inputDim, int hiddenDim, int sequenceLength)
: inputDim_(inputDim), hiddenDim_(hiddenDim), sequenceLength_(sequenceLength),
bufferW_xh_(nullptr), bufferW_hh_(nullptr), bufferBias_(nullptr),
forwardPipelineState_(nullptr), backwardPipelineState_(nullptr)
{}

RNNLayer::~RNNLayer() {
    // Release all buffers
}

void RNNLayer::buildBuffers(MTL::Device* device) {
    float scale = 0.01f;
    
    // Allocate weight buffer: W_xh (inputDim x hiddenDim)
    bufferW_xh_ = device->newBuffer(inputDim_ * hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w_xh = static_cast<float*>(bufferW_xh_->contents());
    for (int i = 0; i < inputDim_ * hiddenDim_; i++)
        w_xh[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    bufferW_xh_->didModifyRange(NS::Range(0, bufferW_xh_->length()));
    
    // Allocate weight buffer: W_hh (hiddenDim x hiddenDim)
    bufferW_hh_ = device->newBuffer(hiddenDim_ * hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* w_hh = static_cast<float*>(bufferW_hh_->contents());
    for (int i = 0; i < hiddenDim_ * hiddenDim_; i++)
        w_hh[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    bufferW_hh_->didModifyRange(NS::Range(0, bufferW_hh_->length()));
    
    // Allocate bias buffer
    bufferBias_ = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    memset(b, 0, hiddenDim_ * sizeof(float));
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    // Allocate per-timestep hidden state buffers
    bufferHiddenStates_.resize(sequenceLength_);
    bufferHiddenPrevStates_.resize(sequenceLength_);
    bufferInputs_.resize(sequenceLength_);
    bufferDenseErrors_.resize(sequenceLength_);
    bufferErrors_.resize(sequenceLength_);
    
    for (int t = 0; t < sequenceLength_; t++) {
        bufferHiddenStates_[t] = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        bufferHiddenPrevStates_[t] = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        bufferErrors_[t] = device->newBuffer(hiddenDim_ * sizeof(float), MTL::ResourceStorageModeManaged);
        
        memset(bufferHiddenStates_[t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(bufferHiddenPrevStates_[t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(bufferErrors_[t]->contents(), 0, hiddenDim_ * sizeof(float));
        
        bufferHiddenStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        bufferHiddenPrevStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        bufferErrors_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        
        // Inputs and dense error buffers are set externally later:
        bufferInputs_[t] = nullptr;
        bufferDenseErrors_[t] = nullptr;
    }
}


void RNNLayer::forward(MTL::CommandBuffer* cmdBuf) {
    for (int t = 0; t < sequenceLength_; ++t) {
#ifdef DEBUG_NETWORK
        float* x = static_cast<float*>(bufferInputs_[t]->contents());
        float* w_xh = static_cast<float*>(bufferW_xh_->contents());
        
        float sumCheck = 0.0f;
        for (int i = 0; i < inputDim_; ++i)
            sumCheck += x[i] * w_xh[i]; // Simple check for first neuron input sum
        
        std::cout << "Pre-activation sum at timestep " << t << ": " << sumCheck << std::endl;
#endif
        
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer(t == 0 ? bufferHiddenPrevStates_[0] : bufferHiddenStates_[t-1], 0, 1);
        encoder->setBuffer(bufferHiddenStates_[t], 0, 2);
        encoder->setBuffer(bufferW_xh_, 0, 3);
        encoder->setBuffer(bufferW_hh_, 0, 4);
        encoder->setBuffer(bufferBias_, 0, 5);
        encoder->setBytes(&inputDim_, sizeof(int), 6);
        encoder->setBytes(&hiddenDim_, sizeof(int), 7);
        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1), MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
    }
}

void RNNLayer::backward(MTL::CommandBuffer* cmdBuf) {
    for (int t = sequenceLength_ - 1; t >= 0; --t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer(t == 0 ? bufferHiddenPrevStates_[0] : bufferHiddenStates_[t-1], 0, 1);
        encoder->setBuffer(bufferW_xh_, 0, 2);
        encoder->setBuffer(bufferW_hh_, 0, 3);
        encoder->setBuffer(bufferBias_, 0, 4);
        encoder->setBuffer(bufferHiddenStates_[t], 0, 5);
        encoder->setBuffer(bufferDenseErrors_[t], 0, 6);
        encoder->setBuffer(bufferErrors_[t], 0, 7);
        encoder->setBytes(&inputDim_, sizeof(int), 8);
        encoder->setBytes(&hiddenDim_, sizeof(int), 9);
        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1), MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
    }
}

// Build pipeline states (forward and backward kernels)
void RNNLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFunction = library->newFunction(NS::String::string("forward_rnn", NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        printf("Error creating forwardPipelineState_: %s\n", error->localizedDescription()->utf8String());
        assert(false);
    }
    forwardFunction->release();
    
    auto backwardFunction = library->newFunction(NS::String::string("learn_rnn", NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        printf("Error creating backwardPipelineState_: %s\n", error->localizedDescription()->utf8String());
        assert(false);
    }
    backwardFunction->release();
}

// Sets input buffer at specific timestep
void RNNLayer::setInputBufferAt(int timestep, MTL::Buffer* inputBuffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferInputs_[timestep] = inputBuffer;
}

// Sets dense layer error buffer at specific timestep
void RNNLayer::setDenseErrorBuffer(MTL::Buffer* denseErrorBuffer, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferDenseErrors_[timestep] = denseErrorBuffer;
}

// Gets the output buffer (hidden state) at specific timestep
MTL::Buffer* RNNLayer::getOutputBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferHiddenStates_[timestep];
}

MTL::Buffer* RNNLayer::getErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferErrors_[timestep];
}
