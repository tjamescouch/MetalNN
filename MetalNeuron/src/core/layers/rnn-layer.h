#ifndef RNNLAYER_H
#define RNNLAYER_H

#include "layer.h"
#include "optimizer.h"
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
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;
    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    void connectInputBuffers(Layer* previousLayer, InputLayer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override;
    
    
    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;
    
    int outputSize() const override;
    void updateTargetBufferAt(const float* targetData, int timestep) override;
    
    void shiftHiddenStates();
    
    void onForwardComplete() override {
        shiftHiddenStates();
    };
    
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    
    void debugLog() override;
    

private:
    int inputDim_;
    int hiddenDim_;
    int sequenceLength_;
    
    ActivationFunction activation_;

    // Explicit mapping of BufferType to buffer arrays
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    MTL::Buffer* bufferWeightGradients_;
    
    // Adam buffers for W_xh (input-to-hidden)
    MTL::Buffer* bufferM_xh_;
    MTL::Buffer* bufferV_xh_;

    // Adam buffers for W_hh (hidden-to-hidden)
    MTL::Buffer* bufferM_hh_;
    MTL::Buffer* bufferV_hh_;
    
    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;

    // Optimizers for different parameter matrices
    std::unique_ptr<Optimizer> optimizerInput_;
    std::unique_ptr<Optimizer> optimizerHidden_;
    std::unique_ptr<Optimizer> optimizerBias_;

    MTL::Buffer* zeroBuffer_;
};

#endif
