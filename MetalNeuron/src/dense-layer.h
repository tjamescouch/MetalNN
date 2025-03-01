#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "layer.h"
#include "data-source.h"
#include <vector>

namespace MTL { class Device; class Buffer; class CommandBuffer; class ComputePipelineState; class Library; }

class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim, int sequenceLength);
    ~DenseLayer();

    void buildPipeline(MTL::Device*, MTL::Library*) override;
    void buildBuffers(MTL::Device*) override;

    void forward(MTL::CommandBuffer*) override;
    void backward(MTL::CommandBuffer*) override;

    void setInputBufferAt(int timestep, MTL::Buffer*);
    void updateTargetBufferAt(DataSource&, int timestep);
    MTL::Buffer* getErrorBufferAt(int timestep) const;
    MTL::Buffer* getOutputBufferAt(int timestep) const;

private:
    int inputDim_, outputDim_, sequenceLength_;
    std::vector<MTL::Buffer*> bufferInputs_, bufferOutputs_, bufferErrors_, bufferTargets_;
    MTL::Buffer *bufferWeights_, *bufferBias_, *bufferDecay_;
    MTL::ComputePipelineState *forwardPipelineState_, *backwardPipelineState_;
};
#endif
