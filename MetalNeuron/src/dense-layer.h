#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "data-source.h"
#include <Metal/Metal.hpp>
#include <vector>

class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim, int sequenceLength);
    ~DenseLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;

    void setInputBufferAt(int timestep, MTL::Buffer* inputBuffer) override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;

    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    MTL::Buffer* getOutputBufferAt(int timestep) const override;
    MTL::Buffer* getErrorBufferAt(int timestep) const override;
    int outputSize() const override;
    
    void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputErrorBufferAt(int timestep) const override;

private:
    int inputDim_;
    int outputDim_;
    int sequenceLength_;

    std::vector<MTL::Buffer*> bufferInputs_;
    std::vector<MTL::Buffer*> bufferOutputs_;
    std::vector<MTL::Buffer*> bufferTargets_;
    std::vector<MTL::Buffer*> bufferErrors_;
    std::vector<MTL::Buffer*> bufferInputErrors_;
    std::vector<MTL::Buffer*> bufferOutputErrors_;

    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;
    

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // DENSE_LAYER_H
