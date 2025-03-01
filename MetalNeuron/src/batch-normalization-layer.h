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
    BatchNormalizationLayer(int featureDim, int sequenceLength, float epsilon = 1e-5f);
    ~BatchNormalizationLayer() override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    void setInputBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(int timestep) const override;

    void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputErrorBufferAt(int timestep) const override;
    
    int outputSize() const override;
    MTL::Buffer* getErrorBufferAt(int timestep) const override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;

private:
    int featureDim_;
    int sequenceLength_;
    float epsilon_;

    // Parameter buffers
    MTL::Buffer* bufferGamma_; // Scale
    MTL::Buffer* bufferBeta_;  // Shift

    // Running averages for inference
    MTL::Buffer* bufferRunningMean_;
    MTL::Buffer* bufferRunningVariance_;

    // Intermediate buffers
    std::vector<MTL::Buffer*> bufferInputs_;
    std::vector<MTL::Buffer*> bufferOutputs_;
    std::vector<MTL::Buffer*> bufferInputErrors_;
    std::vector<MTL::Buffer*> bufferOutputErrors_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;

    void initializeParameters(MTL::Device* device);
};
