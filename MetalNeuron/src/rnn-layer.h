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
    virtual ~RNNLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;

    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    MTL::Buffer* getOutputBufferAt(int timestep) const override;
    MTL::Buffer* getErrorBufferAt(int timestep) const override;
    void setInputBufferAt(int timestep, MTL::Buffer* inputBuffer) override;
    void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputErrorBufferAt(int timestep) const override;
    
    int outputSize() const override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;

    // Shifts stored RNN states forward by one step. (Kept as-is)
    void shiftHiddenStates();
    

private:
    int inputDim_;
    int hiddenDim_;
    int sequenceLength_;
    
    ActivationFunction activation_;

    std::vector<MTL::Buffer*> bufferInputs_;
    std::vector<MTL::Buffer*> bufferHiddenStates_;
    std::vector<MTL::Buffer*> bufferHiddenPrevStates_;
    std::vector<MTL::Buffer*> bufferErrors_;
    std::vector<MTL::Buffer*> bufferDenseErrors_;

    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;

    MTL::Buffer* zeroBuffer_; // CHANGED: holds zero for next_hidden_error boundary
};

#endif
