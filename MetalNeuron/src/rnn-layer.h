#ifndef RNNLAYER_H
#define RNNLAYER_H

#include "layer.h"
#include <vector>

namespace MTL {
    class Device;
    class Buffer;
    class ComputePipelineState;
    class CommandBuffer;
}

class RNNLayer : public Layer {
public:
    RNNLayer(int inputDim, int hiddenDim, int sequenceLength, ActivationFunction activation);
    ~RNNLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;

    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const override;
    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const override;
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override;
    
    int outputSize() const override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;

    // Shifts stored RNN states forward by one step. (Kept as-is)
    void shiftHiddenStates();
    
    void onForwardComplete() override {
        shiftHiddenStates();
    };
    
    void onBackwardComplete() override {
        shiftHiddenStates();
    };
    
    void debugLog() override {
#ifdef DEBUG_RNN_LAYER
        float* weights = static_cast<float*>(bufferW_xh_->contents());
        printf("[RNNLayer DebugLog] bufferW_xh_ sample: %f, %f, %f\n", weights[0], weights[1], weights[2]);
        
        float* weights2 = static_cast<float*>(bufferW_hh_->contents());
        printf("[RNNLayer DebugLog] bufferW_hh_ sample: %f, %f, %f\n", weights2[0], weights2[1], weights2[2]);
        
        // Optionally log biases or other important states:
        float* biases = static_cast<float*>(bufferBias_->contents());
        printf("[RNNLayer DebugLog] bufferBias_ sample: %f, %f, %f\n", biases[0], biases[1], biases[2]);
        
        for (int t = 0; t < sequenceLength_; t++) {
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[RNNLayer DebugLog] outputs at timestep %d: %f, %f, %f\n",
                   t, outputs[0], outputs[1], outputs[2]);
            
            float* prev_outputs = static_cast<float*>(inputBuffers_[BufferType::PrevHiddenState][t]->contents());
            printf("[RNNLayer DebugLog] prev outputs at timestep %d: %f, %f, %f\n",
                   t, prev_outputs[0], prev_outputs[1], prev_outputs[2]);

            float* inputs = static_cast<float*>(inputBuffers_[BufferType::Input][t]->contents());
            printf("[RNNLayer DebugLog] inputs at timestep %d: %f, %f, %f\n",
                   t, inputs[0], inputs[1], inputs[2]);
        }
#endif
    }
    

private:
    int inputDim_;
    int hiddenDim_;
    int sequenceLength_;
    
    ActivationFunction activation_;

    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    // Explicit mapping of BufferType to buffer arrays
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;

    MTL::Buffer* zeroBuffer_; // CHANGED: holds zero for next_hidden_error boundary
};

#endif
