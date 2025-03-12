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
#include "math-lib.h"
#include "logger.h"

BatchNormalizationLayer::BatchNormalizationLayer(int inputDim, int outputDim, int batchSize, int _unused, float learningRate, float epsilon)
: inputDim_(inputDim),
outputDim_(outputDim),
batchSize_(batchSize),
learningRate_(learningRate),
sequenceLength_(1),
isTerminal_(false),
epsilon_(epsilon),
bufferDebug_(nullptr),
bufferGamma_(nullptr),
bufferBeta_(nullptr),
bufferSavedMean_(nullptr),
bufferSavedVariance_(nullptr),
bufferRunningMean_(nullptr),
bufferRunningVariance_(nullptr),
forwardPipelineState_(nullptr),
backwardPipelineState_(nullptr) {
    assert(inputDim == outputDim);
    assert(_unused==1);
    inputBuffers_[BufferType::Input].resize(sequenceLength_, nullptr);
    outputBuffers_[BufferType::Output].resize(sequenceLength_, nullptr);
    bufferSize_ = batchSize_ * outputDim_ * sizeof(float);
}

BatchNormalizationLayer::~BatchNormalizationLayer() {
    if (bufferDebug_) bufferDebug_->release();
    if (bufferGamma_) bufferGamma_->release();
    if (bufferBeta_) bufferBeta_->release();
    if (bufferRunningMean_) bufferRunningMean_->release();
    if (bufferRunningVariance_) bufferRunningVariance_->release();
    if (bufferSavedMean_) {
        bufferSavedMean_->release();
        bufferSavedMean_ = nullptr;
    }
    if (bufferSavedVariance_) {
        bufferSavedVariance_->release();
        bufferSavedVariance_ = nullptr;
    }
    
    for (int t = 0; t < sequenceLength_; ++t) {
        for (auto ob : outputBuffers_) {
            ob.second[t]->release();
        }
    }
    
    if (forwardPipelineState_) forwardPipelineState_->release();
    if (backwardPipelineState_) backwardPipelineState_->release();
}

void BatchNormalizationLayer::initializeParameters(MTL::Device* device) {
    std::vector<float> debug(outputDim_, 0.0f);
    std::vector<float> gamma(outputDim_, 1.0f);
    std::vector<float> beta(outputDim_, 0.0f);
    std::vector<float> runningMean(outputDim_, 0.0f);
    std::vector<float> runningVariance(outputDim_, 1.0f);

    // This is the size for the per-feature arrays:
    size_t bufferSize = sizeof(float) * outputDim_;

    bufferDebug_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferGamma_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferBeta_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferRunningMean_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferRunningVariance_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    // NEW: Buffers for storing exact batch stats during forward, for the backward pass
    bufferSavedMean_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);
    bufferSavedVariance_ = device->newBuffer(bufferSize, MTL::ResourceStorageModeManaged);

    // Initialize all to zeros or ones as appropriate:
    memcpy(bufferDebug_->contents(), debug.data(), bufferSize);
    memcpy(bufferGamma_->contents(), gamma.data(), bufferSize);
    memcpy(bufferBeta_->contents(), beta.data(), bufferSize);
    memcpy(bufferRunningMean_->contents(), runningMean.data(), bufferSize);
    memcpy(bufferRunningVariance_->contents(), runningVariance.data(), bufferSize);

    // Also zero out the "saved" stats buffers so they start in a known state
    std::vector<float> zeros(outputDim_, 0.0f);
    memcpy(bufferSavedMean_->contents(), zeros.data(), bufferSize);
    memcpy(bufferSavedVariance_->contents(), zeros.data(), bufferSize);

    // Mark them as modified
    bufferDebug_->didModifyRange(NS::Range(0, bufferSize));
    bufferGamma_->didModifyRange(NS::Range(0, bufferSize));
    bufferBeta_->didModifyRange(NS::Range(0, bufferSize));
    bufferRunningMean_->didModifyRange(NS::Range(0, bufferSize));
    bufferRunningVariance_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedMean_->didModifyRange(NS::Range(0, bufferSize));
    bufferSavedVariance_->didModifyRange(NS::Range(0, bufferSize));
}

void BatchNormalizationLayer::buildBuffers(MTL::Device* device) {
    initializeParameters(device);
    
    inputBuffers_[BufferType::Input].clear();
    outputBuffers_[BufferType::Output].clear();
    inputBuffers_[BufferType::InputErrors].clear();
    outputBuffers_[BufferType::OutputErrors].clear();
    
    inputBuffers_[BufferType::Input].push_back(device->newBuffer(bufferSize_, MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::Output].push_back(device->newBuffer(bufferSize_, MTL::ResourceStorageModeManaged));
    inputBuffers_[BufferType::InputErrors].push_back(device->newBuffer(bufferSize_, MTL::ResourceStorageModeManaged));
    outputBuffers_[BufferType::OutputErrors].push_back(device->newBuffer(bufferSize_, MTL::ResourceStorageModeManaged));
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

void BatchNormalizationLayer::forward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    bool isTraining = TrainingManager::instance().isTraining();
    
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(forwardPipelineState_);
    
    encoder->setBuffer(inputBuffers_[BufferType::Input][0], 0, 0);        // input
    encoder->setBuffer(outputBuffers_[BufferType::Output][0], 0, 1);      // output
    encoder->setBuffer(bufferGamma_, 0, 2);                               // gamma
    encoder->setBuffer(bufferBeta_, 0, 3);                                // beta
    encoder->setBuffer(bufferRunningMean_, 0, 4);                         // runningMean
    encoder->setBuffer(bufferRunningVariance_, 0, 5);                     // runningVariance

    // NEW: savedMean, savedVariance at indices 6, 7
    encoder->setBuffer(bufferSavedMean_, 0, 6);
    encoder->setBuffer(bufferSavedVariance_, 0, 7);

    // SHIFT the next arguments: epsilon->8, featureDim->9, isTraining->10, batchSize->11, debug->12
    encoder->setBytes(&epsilon_, sizeof(float), 8);
    encoder->setBytes(&outputDim_, sizeof(int), 9);
    encoder->setBytes(&isTraining, sizeof(bool), 10);
    encoder->setBytes(&batchSize, sizeof(uint), 11);
    encoder->setBuffer(bufferDebug_, 0, 12);

    MTL::Size threadsPerGroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
    encoder->endEncoding();
}

void BatchNormalizationLayer::backward(MTL::CommandBuffer* cmdBuf, int batchSize)
{
    bool isTraining = TrainingManager::instance().isTraining();
    
    auto encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(backwardPipelineState_);
    
    // indices:
    encoder->setBuffer(inputBuffers_[BufferType::Input][0],       0, 0); // input
    encoder->setBuffer(inputBuffers_[BufferType::InputErrors][0], 0, 1); // inputErrors
    encoder->setBuffer(outputBuffers_[BufferType::OutputErrors][0], 0, 2); // outputErrors
    encoder->setBuffer(bufferGamma_, 0, 3);     // gamma
    encoder->setBuffer(bufferBeta_, 0, 4);      // beta

    // NEW: savedMean=5, savedVariance=6
    encoder->setBuffer(bufferSavedMean_, 0, 5);
    encoder->setBuffer(bufferSavedVariance_, 0, 6);

    // SHIFT the running stats: 7,8
    encoder->setBuffer(bufferRunningMean_, 0, 7);
    encoder->setBuffer(bufferRunningVariance_, 0, 8);

    // SHIFT the rest:
    encoder->setBytes(&epsilon_,       sizeof(float), 9);    // 9: epsilon
    encoder->setBytes(&outputDim_,     sizeof(int),   10);   // 10: featureDim
    encoder->setBytes(&isTraining,     sizeof(bool),  11);   // 11: isTraining
    encoder->setBytes(&batchSize,      sizeof(uint),  12);   // 12: batchSize
    encoder->setBytes(&learningRate_,  sizeof(float), 13);   // 13: learningRate
    encoder->setBuffer(bufferDebug_,   0,             14);   // 14: debug

    MTL::Size threadsPerGroup = MTL::Size(std::min(outputDim_, 1024), 1, 1);
    MTL::Size threadgroups = MTL::Size((outputDim_ + 1023) / 1024, 1, 1);
    encoder->dispatchThreadgroups(threadgroups, threadsPerGroup);
    encoder->endEncoding();
}

int BatchNormalizationLayer::outputSize() const {
    return outputDim_;
}

void BatchNormalizationLayer::updateTargetBufferAt(const float* targetData, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
}

void BatchNormalizationLayer::updateTargetBufferAt(const float* targetData, int timestep, int batchSize) {
    assert(timestep==0 && "Timesteps not supported for this layer");
}


void BatchNormalizationLayer::setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    assert(buffer && "Setting input buffer to NULL");
    inputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getOutputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    return outputBuffers_[type][timestep];
}

void BatchNormalizationLayer::setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    outputBuffers_[type][timestep] = buffer;
}

MTL::Buffer* BatchNormalizationLayer::getInputBufferAt(BufferType type, int timestep) {
    assert(timestep==0 && "Timesteps not supported for this layer");
    return inputBuffers_[type][timestep];
}

void BatchNormalizationLayer::connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                                  MTL::Buffer* zeroBuffer, int timestep) {
    setInputBufferAt(BufferType::Input, timestep,
                     previousLayer
                     ? previousLayer->getOutputBufferAt(BufferType::Output, timestep)
                     : inputLayer->getOutputBufferAt(BufferType::Output, timestep)
                     );
}

void BatchNormalizationLayer::connectBackwardConnections(Layer* prevLayer, Layer* inputLayer,
                                                  MTL::Buffer* zeroBuffer, int timestep) {
    prevLayer->setInputBufferAt(BufferType::InputErrors, 0, getOutputBufferAt(BufferType::OutputErrors, timestep));
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

void BatchNormalizationLayer::onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
#ifdef F
    Logger::instance().printFloatBuffer(bufferDebug_, "F: Debug", 128);
    

    Logger::instance().printFloatBuffer(bufferBeta_, "Beta", 10);
    Logger::instance().printFloatBuffer(bufferGamma_, "Gamma", 10);
    

    Logger::instance().printFloatBuffer(inputBuffers_[BufferType::Input][0], "[B: Batch Normalization Input]", 10);
    Logger::instance().printFloatBuffer(outputBuffers_[BufferType::Output][0], "[B: Batch Normalization Output]", 10);
    Logger::instance().printFloatBuffer(inputBuffers_[BufferType::InputErrors][0], "[B: Batch Normalization Input Errors]", 10);
    Logger::instance().printFloatBuffer(outputBuffers_[BufferType::OutputErrors][0], "[B: Batch Normalization Output Errors]", 10);
#endif
}

void BatchNormalizationLayer::onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) {
    //Logger::instance().printFloatBuffer(bufferDebug_, "B: Debug", 4);
    bufferBeta_->didModifyRange(NS::Range(0, sizeof(float) * outputDim_));
    //bufferGamma_->didModifyRange(NS::Range(0, sizeof(float) * outputDim_));
    bufferRunningMean_->didModifyRange(NS::Range(0, sizeof(float) * outputDim_));
    bufferRunningVariance_->didModifyRange(NS::Range(0, sizeof(float) * outputDim_));
}
