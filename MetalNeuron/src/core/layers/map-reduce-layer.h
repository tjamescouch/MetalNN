//
//  map-reduce-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#ifndef MAP_REDUCE_LAYER_H
#define MAP_REDUCE_LAYER_H

#include "layer.h"

class MapReduceLayer : public Layer {
public:
    MapReduceLayer(int inputSize, ReductionType reductionType);
    ~MapReduceLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;

    int getSequenceLength() override;
    
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;
    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;

    int outputSize() const override;

    void updateTargetBufferAt(const float* targetData, int timestep) override;

    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;

    void debugLog() override;

    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;

    void saveParameters(std::ostream& outStream) const override;
    void loadParameters(std::istream& inStream) override;

    void setIsTerminal(bool isTerminal) override;

private:
    int output_dim_;
    int inputSize_;
    int sequenceLength_;
    ReductionType reductionType_;
    bool isTerminal_;
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    // Explicit mapping of BufferType to buffer arrays
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
};

#endif
