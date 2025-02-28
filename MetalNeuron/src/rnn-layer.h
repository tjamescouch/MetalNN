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
    class CommandBuffer;
}

class RNNLayer : public Layer {
public:
    RNNLayer(int inputDim, int hiddenDim);
    virtual ~RNNLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    MTL::Buffer* getErrorBuffer() const;
    MTL::Buffer* getOutputBuffer() const;
    void setInputBuffer(MTL::Buffer* inputBuffer);

    // NEW: Setter to bind DenseLayer's error buffer.
    void setDenseErrorBuffer(MTL::Buffer* denseErrorBuffer);

private:
    int inputDim_;
    int hiddenDim_;

    MTL::Buffer* bufferInput_;
    MTL::Buffer* bufferHidden_;
    MTL::Buffer* bufferHiddenPrev_;
    MTL::Buffer* bufferW_xh_;
    MTL::Buffer* bufferW_hh_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferError_;
    MTL::Buffer* bufferDenseError_;  // New buffer for DenseLayer error propagation

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // RNNLAYER_H
