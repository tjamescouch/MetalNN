#include <iostream>
#include <cstring>

#include "rnn-layer.h"
#include "common.h"
#include "weight-initializer.h"

RNNLayer::RNNLayer(int inputDim, int hiddenDim, int sequenceLength, ActivationFunction activation)
: inputDim_(inputDim),
  hiddenDim_(hiddenDim),
  sequenceLength_(sequenceLength),
  bufferW_xh_(nullptr),
  bufferW_hh_(nullptr),
  bufferBias_(nullptr),
  bufferDecay_(nullptr),
  forwardPipelineState_(nullptr),
  backwardPipelineState_(nullptr),
  zeroBuffer_(nullptr),
  activation_(activation)
{}

RNNLayer::~RNNLayer() {
    // Release all allocated buffers and pipeline states
    for (auto buf : bufferInputs_)          if (buf) buf->release();
    for (auto buf : bufferHiddenStates_)    if (buf) buf->release();
    for (auto buf : bufferHiddenPrevStates_)if (buf) buf->release();
    for (auto buf : bufferErrors_)          if (buf) buf->release();
    for (auto buf : bufferDenseErrors_)     if (buf) buf->release();

    if (bufferW_xh_) bufferW_xh_->release();
    if (bufferW_hh_) bufferW_hh_->release();
    if (bufferBias_) bufferBias_->release();
    if (bufferDecay_) bufferDecay_->release();

    if (forwardPipelineState_)  forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();

    if (zeroBuffer_) zeroBuffer_->release(); // CHANGED
}

void RNNLayer::buildBuffers(MTL::Device* device) {
    float decay = 1.0f;
    
    // Allocate weight buffer: W_xh (inputDim x hiddenDim)
    bufferW_xh_ = device->newBuffer(inputDim_ * hiddenDim_ * sizeof(float),
                                    MTL::ResourceStorageModeManaged);
    float* w_xh = static_cast<float*>(bufferW_xh_->contents());
    WeightInitializer::initializeXavier(w_xh, inputDim_, hiddenDim_);
    bufferW_xh_->didModifyRange(NS::Range(0, bufferW_xh_->length()));

    // Allocate decay buffer
    bufferDecay_ = device->newBuffer(sizeof(float), MTL::ResourceStorageModeManaged);
    memcpy(bufferDecay_->contents(), &decay, sizeof(float));
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));

    // Allocate weight buffer: W_hh (hiddenDim x hiddenDim)
    bufferW_hh_ = device->newBuffer(hiddenDim_ * hiddenDim_ * sizeof(float),
                                    MTL::ResourceStorageModeManaged);
    float* w_hh = static_cast<float*>(bufferW_hh_->contents());
    WeightInitializer::initializeXavier(w_hh, hiddenDim_, hiddenDim_);
    bufferW_hh_->didModifyRange(NS::Range(0, bufferW_hh_->length()));

    // Allocate bias buffer
    bufferBias_ = device->newBuffer(hiddenDim_ * sizeof(float),
                                    MTL::ResourceStorageModeManaged);
    float* b = static_cast<float*>(bufferBias_->contents());
    WeightInitializer::initializeBias(b, hiddenDim_);
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    // Allocate per-timestep hidden states, error, etc.
    bufferHiddenStates_.resize(sequenceLength_);
    bufferHiddenPrevStates_.resize(sequenceLength_);
    bufferInputs_.resize(sequenceLength_);
    bufferDenseErrors_.resize(sequenceLength_);
    bufferErrors_.resize(sequenceLength_);
    
    for (int t = 0; t < sequenceLength_; t++) {
        bufferHiddenStates_[t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                  MTL::ResourceStorageModeManaged);
        bufferHiddenPrevStates_[t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                       MTL::ResourceStorageModeManaged);
        bufferErrors_[t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                             MTL::ResourceStorageModeManaged);

        memset(bufferHiddenStates_[t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(bufferHiddenPrevStates_[t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(bufferErrors_[t]->contents(), 0, hiddenDim_ * sizeof(float));

        bufferHiddenStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        bufferHiddenPrevStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        bufferErrors_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));

        // Set these to nullptr initially; assigned externally
        bufferInputs_[t] = nullptr;
        bufferDenseErrors_[t] = nullptr;
    }

    // CHANGED: Allocate zero buffer for next_hidden_error at boundary t = sequenceLength_-1
    zeroBuffer_ = device->newBuffer(hiddenDim_ * sizeof(float),
                                    MTL::ResourceStorageModeManaged);
    memset(zeroBuffer_->contents(), 0, hiddenDim_ * sizeof(float));
    zeroBuffer_->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
}

void RNNLayer::forward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);
    
    for (int t = 0; t < sequenceLength_; ++t) {
#ifdef DEBUG_NETWORK
        float* x = static_cast<float*>(bufferInputs_[t]->contents());
        float* w_xh = static_cast<float*>(bufferW_xh_->contents());
        float* pDecay = static_cast<float*>(bufferDecay_->contents());
        
        float sumCheck = 0.0f;
        for (int i = 0; i < inputDim_; ++i)
            sumCheck += x[i] * w_xh[i]; // Simple check
        std::cout << "Pre-activation sum at timestep " << t << ": " << sumCheck << std::endl;
        std::cout << "RNN decay " << *pDecay << std::endl;
#endif
        
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);

        // For the first timestep, use bufferHiddenPrevStates_[0];
        // for subsequent timesteps, use bufferHiddenStates_[t-1].
        encoder->setBuffer(bufferInputs_[t],               0, 0);
        encoder->setBuffer((t == 0
                            ? bufferHiddenPrevStates_[0]
                            : bufferHiddenStates_[t-1]),
                           0, 1);
        encoder->setBuffer(bufferHiddenStates_[t],         0, 2);
        encoder->setBuffer(bufferW_xh_,                    0, 3);
        encoder->setBuffer(bufferW_hh_,                    0, 4);
        encoder->setBuffer(bufferBias_,                    0, 5);
        encoder->setBytes(&inputDim_,       sizeof(int),      6);
        encoder->setBytes(&hiddenDim_,      sizeof(int),      7);
        encoder->setBuffer(bufferDecay_,                   0, 8);
        encoder->setBytes(&activationRaw, sizeof(uint),       9);

        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1),
                                 MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
    }
}

void RNNLayer::backward(MTL::CommandBuffer* cmdBuf) {
    uint activationRaw = static_cast<uint>(activation_);
    
    for (int t = sequenceLength_ - 1; t >= 0; --t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);

        // buffers:
        encoder->setBuffer(bufferInputs_[t], 0, 0);
        encoder->setBuffer((t == 0
                            ? bufferHiddenPrevStates_[0]
                            : bufferHiddenStates_[t - 1]),
                           0, 1);
        encoder->setBuffer(bufferW_xh_, 0, 2);
        encoder->setBuffer(bufferW_hh_, 0, 3);
        encoder->setBuffer(bufferBias_, 0, 4);
        encoder->setBuffer(bufferHiddenStates_[t], 0, 5);
        encoder->setBuffer(bufferDenseErrors_[t], 0, 6);

        // CHANGED: For t=sequenceLength_-1, no next timestep => pass zeroBuffer_.
        // Otherwise, pass bufferErrors_[t+1].
        if (t == sequenceLength_ - 1) {
            encoder->setBuffer(zeroBuffer_, 0, 7); // no next-hidden-error at the last
        } else {
            encoder->setBuffer(bufferErrors_[t + 1], 0, 7);
        }

        // Our own hidden error at this timestep
        encoder->setBuffer(bufferErrors_[t], 0, 8);

        encoder->setBytes(&inputDim_,  sizeof(int), 9);
        encoder->setBytes(&hiddenDim_, sizeof(int), 10);
        encoder->setBuffer(bufferDecay_, 0, 11);
        encoder->setBytes(&activationRaw, sizeof(uint),       12);

        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1),
                                 MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
    }
}

void RNNLayer::buildPipeline(MTL::Device* device, MTL::Library* library) {
    NS::Error* error = nullptr;
    
    auto forwardFunction = library->newFunction(NS::String::string("forward_rnn",
                                   NS::UTF8StringEncoding));
    forwardPipelineState_ = device->newComputePipelineState(forwardFunction, &error);
    if (!forwardPipelineState_) {
        printf("Error creating forwardPipelineState_: %s\n",
               error->localizedDescription()->utf8String());
        assert(false);
    }
    forwardFunction->release();
    
    auto backwardFunction = library->newFunction(NS::String::string("learn_rnn",
                                    NS::UTF8StringEncoding));
    backwardPipelineState_ = device->newComputePipelineState(backwardFunction, &error);
    if (!backwardPipelineState_) {
        printf("Error creating backwardPipelineState_: %s\n",
               error->localizedDescription()->utf8String());
        assert(false);
    }
    backwardFunction->release();
}

void RNNLayer::setInputBufferAt(int timestep, MTL::Buffer* inputBuffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferInputs_[timestep] = inputBuffer;
}

/*
void RNNLayer::setDenseErrorBuffer(MTL::Buffer* denseErrorBuffer, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferDenseErrors_[timestep] = denseErrorBuffer;
}*/

MTL::Buffer* RNNLayer::getOutputBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferHiddenStates_[timestep];
}

MTL::Buffer* RNNLayer::getErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferErrors_[timestep];
}

void RNNLayer::shiftHiddenStates() {
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        memcpy(bufferHiddenStates_[t]->contents(),
               bufferHiddenStates_[t+1]->contents(),
               hiddenDim_ * sizeof(float));
        memcpy(bufferHiddenPrevStates_[t]->contents(),
               bufferHiddenPrevStates_[t+1]->contents(),
               hiddenDim_ * sizeof(float));

        bufferHiddenStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        bufferHiddenPrevStates_[t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
    }
    // Preserve continuity in the last slot instead of zeroing
    memcpy(bufferHiddenStates_[sequenceLength_-1]->contents(),
           bufferHiddenStates_[sequenceLength_-2]->contents(),
           hiddenDim_ * sizeof(float));

    memcpy(bufferHiddenPrevStates_[sequenceLength_-1]->contents(),
           bufferHiddenStates_[sequenceLength_-2]->contents(),
           hiddenDim_ * sizeof(float));

    bufferHiddenStates_[sequenceLength_-1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
    bufferHiddenPrevStates_[sequenceLength_-1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));

}

int RNNLayer::outputSize() const {
    return hiddenDim_;
}

void RNNLayer::updateTargetBufferAt(DataSource& targetData, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    
    float* denseErrorData = static_cast<float*>(bufferDenseErrors_[timestep]->contents());
    const float* outputData = static_cast<float*>(bufferHiddenStates_[timestep]->contents());
    const float* target = targetData.get_data_buffer_at(timestep);

    for (int i = 0; i < hiddenDim_; ++i) {
        denseErrorData[i] = outputData[i] - target[i]; // Simple mean-squared error gradient
    }

    bufferDenseErrors_[timestep]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
}

void RNNLayer::setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) {
    assert(timestep >= 0 && timestep < sequenceLength_);
    bufferDenseErrors_[timestep] = buffer;
}

MTL::Buffer* RNNLayer::getInputErrorBufferAt(int timestep) const {
    assert(timestep >= 0 && timestep < sequenceLength_);
    return bufferErrors_[timestep];
}
