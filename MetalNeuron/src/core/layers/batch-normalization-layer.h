//
//  batch-normalization-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once
#include <Metal/Metal.hpp>
#include <vector>
#include "layer.h"

class BatchNormalizationLayer : public Layer {
public:
    BatchNormalizationLayer(int inputDim, int outputDim, int sequenceLength, float epsilon = 1e-5f);
    ~BatchNormalizationLayer() override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override;
    void updateTargetBufferAt(const float* targetData, int timestep) override;
    void updateTargetBufferAt(const float* targetData, int timestep, int batchSize) override;
    
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {};
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void debugLog() override {/*TODO*/}
    
    int getSequenceLength() override { return sequenceLength_; };
    
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
private:
    int inputDim_;
    int outputDim_;
    int sequenceLength_;
    float epsilon_;
    bool isTerminal_;

    // Parameter buffers
    MTL::Buffer* bufferGamma_; // Scale
    MTL::Buffer* bufferBeta_;  // Shift

    // Running averages for inference
    MTL::Buffer* bufferRunningMean_;
    MTL::Buffer* bufferRunningVariance_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;

    void initializeParameters(MTL::Device* device);
};
