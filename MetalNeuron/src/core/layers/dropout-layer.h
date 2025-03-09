//
//  dropout-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once

#include "layer.h"
#include <Metal/Metal.hpp>

class DropoutLayer : public Layer {
public:
    DropoutLayer(float rate, int inputDim, int featureDim, int batchSize, int sequenceLength);
    ~DropoutLayer() override;

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;

    void updateTargetBufferAt(const float*, int) override {}
    void updateTargetBufferAt(const float*, int, int) override {}
    
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override { return featureDim_; }
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {
        for (int t = 0; t < sequenceLength_; ++t)
            memset(outputBuffers_[BufferType::OutputErrors][t]->contents(), 0, outputBuffers_[BufferType::OutputErrors][t]->length());
        
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
    int inputDim_;
    int featureDim_;
    bool isTerminal_;
    int batchSize_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    // New member for CPU-fed randomness
    MTL::Buffer* bufferRandomMask_;
    MTL::Device* _pDevice;
        
    void generateRandomMask();
};
