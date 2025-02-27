#ifndef RNNLAYER_H
#define RNNLAYER_H

#include "layer.h"

// Forward declarations for Metal classes
namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class Buffer;
    class ComputePipelineState;
    class Function;
    class CompileOptions;
    class ComputeCommandEncoder;
}

namespace MTL {
    class Device;
    class CommandBuffer;
}

class RNNLayer : public Layer {
public:
    RNNLayer(int inputDim, int hiddenDim);
    virtual ~RNNLayer();

    // Build the compute pipeline for this layer.
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    // Allocate buffers needed by this layer.
    void buildBuffers(MTL::Device* device) override;
    // Record commands for the forward pass.
    void forward(MTL::CommandBuffer* cmdBuf) override;
    // Record commands for the backward pass.
    void backward(MTL::CommandBuffer* cmdBuf) override;
    
    MTL::Buffer* getErrorBuffer() const;

    // Returns the buffer containing the hidden state (i.e. this layer’s output).
    MTL::Buffer* getOutputBuffer() const;

    // Optionally, you can provide methods to set the input buffer pointer.
    void setInputBuffer(MTL::Buffer* inputBuffer);

private:
    int inputDim_;
    int hiddenDim_;

    // Metal buffers for this layer.
    MTL::Buffer* bufferInput_;      // Pointer to input (set externally)
    MTL::Buffer* bufferHidden_;
    MTL::Buffer* bufferHiddenPrev_;
    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;
    // New error buffer for learning.
    MTL::Buffer* bufferError_;

    // Pipeline states for forward and backward passes.
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // RNNLAYER_H
