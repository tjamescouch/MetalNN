//
//  residual-connection-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-13.
//

#ifndef RESIDUAL_CONNECTION_LAYER_H
#define RESIDUAL_CONNECTION_LAYER_H

#include "layer.h"

class ResidualConnectionLayer : public Layer {
public:
    ResidualConnectionLayer(int featureDim, int batchSize);
    ~ResidualConnectionLayer();

    void forward(MTL::CommandBuffer* commandBuffer, int batchSize) override;
    void backward(MTL::CommandBuffer* commandBuffer, int batchSize) override;

    void buildBuffers(MTL::Device* device) override;
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;

    void setResidualInput(MTL::Buffer* residualBuffer);
    bool supportsResidual() const { return true; }

    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;
    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;

    int inputSize() const override;
    int outputSize() const override;

    void updateTargetBufferAt(const float* targetData, int timestep) override;
    void updateTargetBufferAt(const float* targetData, int timestep, int batchSize) override;

    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                   MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                    MTL::Buffer* zeroBuffer, int timestep) override;

    void debugLog() override;
    void onForwardComplete(MTL::CommandQueue* commandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* commandQueue, int batchSize) override;

    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;

    void setIsTerminal(bool isTerminal) override;

private:
    int featureDim_;
    int batchSize_;
    bool isTerminal_;

    MTL::Buffer* residualInputBuffer_;
    std::vector<MTL::Buffer*> inputBuffers_;
    std::vector<MTL::Buffer*> outputBuffers_;
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // RESIDUAL_CONNECTION_LAYER_H
