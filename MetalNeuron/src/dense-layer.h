// dense-layer.h
#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"
#include "data-source.h"

namespace MTL {
    class Device;
    class Buffer;
    class CommandBuffer;
    class ComputePipelineState;
    class Library;
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
    void setInputBuffer(MTL::Buffer* inputBuffer);

    void updateTargetBuffer(DataSource& ds);

private:
    int inputDim_, outputDim_;
    MTL::Buffer *bufferInput_, *bufferOutput_, *bufferWeights_, *bufferBias_, *bufferYhat_, *bufferError_;
    MTL::ComputePipelineState *forwardPipelineState_, *backwardPipelineState_;
};

#endif // DENSELAYER_H
