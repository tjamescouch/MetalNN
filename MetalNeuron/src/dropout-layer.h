//
//  dropout-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once

#include "layer.h"
#import "data-source.h"
#include <Metal/Metal.hpp>

class DropoutLayer : public Layer {
public:
    DropoutLayer(float rate, int featureDim, int sequenceLength);
    ~DropoutLayer() override;

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    void updateTargetBufferAt(DataSource&, int) override {}
    
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const override;
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    
    int outputSize() const override { return featureDim_; }
    
    void onForwardComplete() override {};
    void onBackwardComplete() override {};
    
    void debugLog() override {
#ifdef DEBUG_DROPOUT_LAYER
        for (int t = 0; t < sequenceLength_; ++t) {
            float* inputs = static_cast<float*>(inputBuffers_[BufferType::Input][t]->contents());
            printf("[DenseLayer Input Debug] timestep %d: ", t);
            for(int i = 0; i < inputBuffers_[BufferType::Input][t]->length()/sizeof(float); ++i)
                printf(" %f, ", inputs[i]);
            printf("\n");
 
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[DenseLayer Input Debug] timestep %d: ", t);
            for(int i = 0; i < outputBuffers_[BufferType::Output][t]->length()/sizeof(float); ++i)
                printf(" %f, ", outputs[i]);
            printf("\n");
        }
#endif
    }
    
private:
    float rate_;
    int sequenceLength_;
    int featureDim_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    // New member for CPU-fed randomness
    MTL::Buffer* bufferRandomMask_;
    
    // Helper to generate CPU-side random mask
    void generateRandomMask(MTL::Device* device);
};
