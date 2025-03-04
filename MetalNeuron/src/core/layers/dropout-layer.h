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

    void updateTargetBufferAt(const float*, int) override {}
    
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    void connectInputBuffers(Layer* previousLayer, InputLayer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    
    int outputSize() const override { return featureDim_; }
    
    
    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;
    
    void onForwardComplete() override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue) override {
        generateRandomMask();
    };
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    int getSequenceLength() override { return sequenceLength_; };
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
    void debugLog() override {
#ifdef DEBUG_DROPOUT_LAYER
        for (int t = 0; t < sequenceLength_; ++t) {
            float* inputs = static_cast<float*>(inputBuffers_[BufferType::Input][t]->contents());
            printf("[DropoutLayer Input Debug] timestep %d: ", t);
            for(int i = 0; i < inputBuffers_[BufferType::Input][t]->length()/sizeof(float); ++i)
                printf(" %f, ", inputs[i]);
            printf("\n");
 
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[DropoutLayer Output Debug] timestep %d: ", t);
            for(int i = 0; i < outputBuffers_[BufferType::Output][t]->length()/sizeof(float); ++i)
                printf(" %f, ", outputs[i]);
            printf("\n");
            
            float* mask = static_cast<float*>( bufferRandomMask_->contents());
            printf("[DropoutLayer Mask Debug] timestep %d: ", t);
            for(int i = 0; i < bufferRandomMask_->length()/sizeof(float); ++i)
                printf(" %f, ", mask[i]);
            printf("\n");
        }
#endif
    }
    
private:
    float rate_;
    int sequenceLength_;
    int featureDim_;
    bool isTerminal_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    // New member for CPU-fed randomness
    MTL::Buffer* bufferRandomMask_;
    MTL::Device* _pDevice;
        
    void generateRandomMask();
};
