#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"
#include "data-source.h"

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

class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim);
    virtual ~DenseLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    MTL::Buffer* getErrorBuffer() const;
    MTL::Buffer* getOutputBuffer() const;
    // Set the input buffer pointer (typically the output from previous layer).
    void setInputBuffer(MTL::Buffer* inputBuffer);

    // Update the target output (y_hat) buffer from a DataSource.
    void updateTargetBuffer(DataSource& ds);

private:
    int inputDim_;
    int outputDim_;

    MTL::Buffer* bufferInput_;
    MTL::Buffer* bufferOutput_;
    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;

    // New buffers for target output and error.
    MTL::Buffer* bufferYhat_;
    MTL::Buffer* bufferError_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // DENSELAYER_H
