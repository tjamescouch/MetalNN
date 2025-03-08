#include <iostream>
#include <cstring>
#include <cmath>

#include "input-layer.h"
#include "rnn-layer.h"
#include "common.h"
#include "weight-initializer.h"
#include "adam-optimizer.h"
#include "configuration-manager.h"

RNNLayer::RNNLayer(int inputDim, int hiddenDim, int sequenceLength, ActivationFunction activation, float learningRate)
: inputDim_(inputDim),
hiddenDim_(hiddenDim),
sequenceLength_(sequenceLength),
bufferWeightGradients_(nullptr),
bufferW_xh_(nullptr),
bufferW_hh_(nullptr),
bufferBias_(nullptr),
bufferDecay_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr),
zeroBuffer_(nullptr),
isTerminal_(false),
activation_(activation),
learningRate_(learningRate)
{
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::PrevHiddenState].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_, nullptr);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::HiddenErrors].resize(sequenceLength_, nullptr);
}

RNNLayer::~RNNLayer() {
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
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
    const int numWeights = inputDim_ * hiddenDim_;
    const int weightBufferSize = numWeights * sizeof(float);
    
    bufferWeightGradients_ = device->newBuffer(weightBufferSize, MTL::ResourceStorageModeManaged);
    
    
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
    outputBuffers_[BufferType::Output].resize(sequenceLength_);
    inputBuffers_[BufferType::PrevHiddenState].resize(sequenceLength_);
    inputBuffers_[BufferType::Input].resize(sequenceLength_);
    inputBuffers_[BufferType::Targets].resize(sequenceLength_);
    outputBuffers_[BufferType::OutputErrors].resize(sequenceLength_);
    outputBuffers_[BufferType::HiddenErrors].resize(sequenceLength_);
    inputBuffers_[BufferType::InputErrors].resize(sequenceLength_);
    
    for (int t = 0; t < sequenceLength_; t++) {
        inputBuffers_[BufferType::Targets][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                      MTL::ResourceStorageModeManaged);
        outputBuffers_[BufferType::Output][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                  MTL::ResourceStorageModeManaged);
        inputBuffers_[BufferType::PrevHiddenState][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                          MTL::ResourceStorageModeManaged);
        outputBuffers_[BufferType::OutputErrors][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                        MTL::ResourceStorageModeManaged);
        inputBuffers_[BufferType::InputErrors][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                        MTL::ResourceStorageModeManaged);
        outputBuffers_[BufferType::HiddenErrors][t] = device->newBuffer(hiddenDim_ * sizeof(float),
                                                                        MTL::ResourceStorageModeManaged);
        
        memset(outputBuffers_[BufferType::Output][t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(inputBuffers_[BufferType::PrevHiddenState][t]->contents(), 0, hiddenDim_ * sizeof(float));
        memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, hiddenDim_ * sizeof(float));
        
        outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        inputBuffers_[BufferType::PrevHiddenState][t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        outputBuffers_[BufferType::OutputErrors][t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        
        inputBuffers_[BufferType::Input][t] = nullptr;
    }
    
    zeroBuffer_ = device->newBuffer(hiddenDim_ * sizeof(float),
                                    MTL::ResourceStorageModeManaged);
    
    optimizerInput_->buildBuffers(device, inputDim_ * hiddenDim_ * sizeof(float));
    optimizerHidden_->buildBuffers(device, hiddenDim_ * hiddenDim_ * sizeof(float));
    optimizerBias_->buildBuffers(device, hiddenDim_ * sizeof(float));

}

void RNNLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    uint activationRaw = static_cast<uint>(activation_);
    
    for (int t = 0; t < sequenceLength_; ++t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(forwardPipelineState_);
        
        encoder->setBuffer(
                           (t == 0
                            ? inputBuffers_[BufferType::Input][0]
                            : outputBuffers_[BufferType::Output][t-1]),
                           0, 0);
        encoder->setBuffer((t == 0
                            ? inputBuffers_[BufferType::PrevHiddenState][0]
                            : outputBuffers_[BufferType::Output][t-1]),
                           0, 1);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t],         0, 2);
        encoder->setBuffer(bufferW_xh_,                    0, 3);
        encoder->setBuffer(bufferW_hh_,                    0, 4);
        encoder->setBuffer(bufferBias_,                    0, 5);
        encoder->setBytes(&inputDim_,       sizeof(int),      6);
        encoder->setBytes(&hiddenDim_,      sizeof(int),      7);
        encoder->setBytes(&activationRaw, sizeof(uint),       8);
        
        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1),
                                 MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
    }
}

void RNNLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize) {
    uint activationRaw = static_cast<uint>(activation_);
    
    for (int t = sequenceLength_ - 1; t >= 0; --t) {
        auto encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(backwardPipelineState_);
        
        // buffers:
        encoder->setBuffer((t == 0
                            ? inputBuffers_[BufferType::Input][0]
                            : outputBuffers_[BufferType::Output][t - 1]),
                           0, 0);
        encoder->setBuffer((t == 0
                            ? inputBuffers_[BufferType::PrevHiddenState][0]
                            : outputBuffers_[BufferType::Output][t - 1]),
                           0, 1);
        encoder->setBuffer(bufferW_xh_, 0, 2);
        encoder->setBuffer(bufferW_hh_, 0, 3);
        encoder->setBuffer(bufferBias_, 0, 4);
        encoder->setBuffer(outputBuffers_[BufferType::Output][t], 0, 5);
        encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][t], 0, 6);
        
        if (t == sequenceLength_ - 1) {
            encoder->setBuffer(zeroBuffer_, 0, 7);
        } else if (isTerminal_) {
            encoder->setBuffer(inputBuffers_[BufferType::Targets][t + 1], 0, 7);
        } else {
            encoder->setBuffer(inputBuffers_[BufferType::InputErrors][t + 1], 0, 7);
        }
        
        // Our own hidden error at this timestep
        encoder->setBuffer(outputBuffers_[BufferType::HiddenErrors][t], 0, 8);
        
        encoder->setBytes(&inputDim_,  sizeof(int), 9);
        encoder->setBytes(&hiddenDim_, sizeof(int), 10);
        encoder->setBuffer(bufferDecay_, 0, 11);
        encoder->setBytes(&activationRaw, sizeof(uint),       12);
        encoder->setBytes(&batchSize, sizeof(uint),       13);
        
        
        encoder->dispatchThreads(MTL::Size(hiddenDim_, 1, 1),
                                 MTL::Size(std::min(hiddenDim_, 1024), 1, 1));
        encoder->endEncoding();
        
        
        if (t > 0) {
            memcpy(outputBuffers_[BufferType::OutputErrors][t - 1]->contents(), outputBuffers_[BufferType::OutputErrors][t]->contents(), hiddenDim_ * sizeof(float));
            outputBuffers_[BufferType::OutputErrors][t - 1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        }
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
    
    ModelConfig* pConfig = ConfigurationManager::instance().getConfig();
    auto parameters = pConfig->training.optimizer.parameters;
    
    float lr      = pConfig->training.optimizer.learning_rate;
    float beta1   = parameters["beta1"].get_value_or<float>(0.9f);
    float beta2   = parameters["beta2"].get_value_or<float>(0.999f);
    float epsilon = parameters["epsilon"].get_value_or<float>(1e-8);
    
    optimizerInput_  = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerHidden_ = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);
    optimizerBias_   = std::make_unique<AdamOptimizer>(lr, beta1, beta2, epsilon);

    optimizerInput_->buildPipeline(device, library);
    optimizerHidden_->buildPipeline(device, library);
    optimizerBias_->buildPipeline(device, library);
}

void RNNLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* RNNLayer::getOutputBufferAt(BufferType type, int timestep) {
    auto it = outputBuffers_.find(type);
    if (it != outputBuffers_.end()) {
        return it->second[timestep];
    }
    return nullptr; // Handle error or return a default buffer explicitly if appropriate.
}


void RNNLayer::shiftHiddenStates() {
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        memcpy(outputBuffers_[BufferType::Output][t]->contents(),
               outputBuffers_[BufferType::Output][t+1]->contents(),
               hiddenDim_ * sizeof(float));
        memcpy(inputBuffers_[BufferType::PrevHiddenState][t]->contents(),
               inputBuffers_[BufferType::PrevHiddenState][t+1]->contents(),
               hiddenDim_ * sizeof(float));
        
        outputBuffers_[BufferType::Output][t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        inputBuffers_[BufferType::PrevHiddenState][t]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
    }
    
    if (sequenceLength_ > 1) {
        // Preserve continuity in the last slot instead of zeroing
        memcpy(outputBuffers_[BufferType::Output][sequenceLength_-1]->contents(),
               outputBuffers_[BufferType::Output][sequenceLength_-2]->contents(),
               hiddenDim_ * sizeof(float));
        
        memcpy(inputBuffers_[BufferType::PrevHiddenState][sequenceLength_-1]->contents(),
               outputBuffers_[BufferType::Output][sequenceLength_-2]->contents(),
               hiddenDim_ * sizeof(float));
        
        outputBuffers_[BufferType::Output][sequenceLength_-1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
        inputBuffers_[BufferType::PrevHiddenState][sequenceLength_-1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
    }
}

int RNNLayer::inputSize() const {
    return inputDim_;
}

int RNNLayer::outputSize() const {
    return hiddenDim_;
}

void RNNLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(timestep >= 0 && timestep < sequenceLength_);

    float* targetBuffer = static_cast<float*>(inputBuffers_[BufferType::Targets][timestep]->contents());
    memcpy(targetBuffer, targetData, hiddenDim_ * sizeof(float));
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
}

void RNNLayer::updateTargetBufferAt(const float* targetData, int timestep, int batchSize) {
    assert(timestep >= 0 && timestep < sequenceLength_);

    float* targetBuffer = static_cast<float*>(inputBuffers_[BufferType::Targets][timestep]->contents());
    memcpy(targetBuffer, targetData, hiddenDim_ * batchSize * sizeof(float));
    inputBuffers_[BufferType::Targets][timestep]->didModifyRange(NS::Range(0, hiddenDim_ * batchSize * sizeof(float)));
}


void RNNLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* RNNLayer::getInputBufferAt(BufferType type, int timestep) {
    return inputBuffers_[type][timestep];
}

void RNNLayer::connectForwardConnections(Layer* prevLayer,
                                   Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    // If there's a previous layer, get its output
    if (prevLayer) {
        setInputBufferAt(BufferType::Input, timestep,
                         prevLayer->getOutputBufferAt(BufferType::Output, prevLayer->getSequenceLength() - timestep - 1)
                         );
    } else {
        // If this is the first layer, read from the InputLayer
        setInputBufferAt(BufferType::Input, timestep,
                         inputLayer->getOutputBufferAt(BufferType::Output, inputLayer->getSequenceLength() - timestep - 1)
                         );
    }
    
    // Also set the previous hidden state
    setInputBufferAt(BufferType::PrevHiddenState, timestep,
                     (timestep == 0)
                     ? zeroBuffer
                     : getOutputBufferAt(BufferType::Output, timestep - 1)
                     );
}

void RNNLayer::connectBackwardConnections(Layer* prevLayer,
                                   Layer* _inputLayer,
                                   MTL::Buffer* zeroBuffer,
                                   int timestep)
{
    prevLayer->setInputBufferAt(BufferType::InputErrors, 0, getOutputBufferAt(BufferType::OutputErrors, timestep));
}

void RNNLayer::saveParameters(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(bufferW_xh_->contents()), bufferW_xh_->length());
    os.write(reinterpret_cast<const char*>(bufferW_hh_->contents()), bufferW_hh_->length());
    os.write(reinterpret_cast<const char*>(bufferBias_->contents()), bufferBias_->length());
    os.write(reinterpret_cast<const char*>(bufferDecay_->contents()), bufferDecay_->length());
}

void RNNLayer::loadParameters(std::istream& is) {
    is.read(reinterpret_cast<char*>(bufferW_xh_->contents()), bufferW_xh_->length());
    bufferW_xh_->didModifyRange(NS::Range(0, bufferW_xh_->length()));
    
    is.read(reinterpret_cast<char*>(bufferW_hh_->contents()), bufferW_hh_->length());
    bufferW_hh_->didModifyRange(NS::Range(0, bufferW_hh_->length()));
    
    is.read(reinterpret_cast<char*>(bufferBias_->contents()), bufferBias_->length());
    bufferBias_->didModifyRange(NS::Range(0, bufferBias_->length()));
    
    is.read(reinterpret_cast<char*>(bufferDecay_->contents()), bufferDecay_->length());
    bufferDecay_->didModifyRange(NS::Range(0, bufferDecay_->length()));
}

void RNNLayer::debugLog() {
#ifdef DEBUG_RNN_LAYER
    float* outputErrors = static_cast<float*>(outputBuffers_[BufferType::OutputErrors][0]->contents());
    size_t outputErrorCount = outputBuffers_[BufferType::OutputErrors][0]->length() / sizeof(float);
    
    float outputErrorNorm = 0.0f;
    for (size_t i = 0; i < outputErrorCount; ++i)
        outputErrorNorm += outputErrors[i] * outputErrors[i];
    
    outputErrorNorm = sqrtf(outputErrorNorm);
    const std::string s = std::string("[") + (isTerminal_ ? "Terminal " : "") +  "RNN Layer DebugLog] Output Error Gradient L2 Norm: %.10e\n";
    printf(s.c_str(), outputErrorNorm);
#endif
}


void RNNLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    shiftHiddenStates();
};

void RNNLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    auto cmdBuf = _pCommandQueue->commandBuffer();
    auto encoder = cmdBuf->computeCommandEncoder();

    optimizerInput_->encode(encoder, bufferW_xh_, inputDim_ * hiddenDim_, batchSize);
    optimizerHidden_->encode(encoder, bufferW_hh_, hiddenDim_ * hiddenDim_, batchSize);
    optimizerBias_->encode(encoder, bufferBias_, hiddenDim_, batchSize);

    encoder->endEncoding();

    //for (int t = 0; t < sequenceLength_; t++) {
    //memset(outputBuffers_[BufferType::OutputErrors][sequenceLength_ - 1]->contents(), 0, outputBuffers_[BufferType::OutputErrors][sequenceLength_ - 1]->length());
    //}
    memset(outputBuffers_[BufferType::OutputErrors][sequenceLength_ - 1]->contents(), 0, outputBuffers_[BufferType::OutputErrors][sequenceLength_ - 1]->length());
    outputBuffers_[BufferType::OutputErrors][sequenceLength_ - 1]->didModifyRange(NS::Range(0, hiddenDim_ * sizeof(float)));
    
    shiftHiddenStates();
}

