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
    RNNLayer(int inputDim, int hiddenDim, int sequenceLength);
    virtual ~RNNLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;

    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    MTL::Buffer* getOutputBufferAt(int timestep) const;
    MTL::Buffer* getErrorBufferAt(int timestep) const;
    void setInputBufferAt(int timestep, MTL::Buffer* inputBuffer);
    void setDenseErrorBuffer(MTL::Buffer* denseErrorBuffer, int timestep);

    // Shifts stored RNN states forward by one step. (Kept as-is)
    void shiftHiddenStates();

private:
    int inputDim_;
    int hiddenDim_;
    int sequenceLength_;

    std::vector<MTL::Buffer*> bufferInputs_;
    std::vector<MTL::Buffer*> bufferHiddenStates_;
    std::vector<MTL::Buffer*> bufferHiddenPrevStates_;
    std::vector<MTL::Buffer*> bufferErrors_;
    std::vector<MTL::Buffer*> bufferDenseErrors_;

    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;

    MTL::Buffer* zeroBuffer_; // CHANGED: holds zero for next_hidden_error boundary
};

#endif
