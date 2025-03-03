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
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const override;
    
    int outputSize() const override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;
    
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override;
    
    
    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;
    
    void onForwardComplete() override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue) override {};
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void debugLog() override {/*TODO*/}
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

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;

    void initializeParameters(MTL::Device* device);
};
