//
//  map-reduce-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#ifndef MAP_REDUCE_LAYER_H
#define MAP_REDUCE_LAYER_H

#include "layer.h"

enum class ReductionType {
    Sum,
    Mean,
    Max,
    Min
};

class MapReduceLayer : public Layer {
public:
    MapReduceLayer(int inputSize, int sequenceLength, ReductionType reductionType);
    ~MapReduceLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

    int getSequenceLength() override;
    
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer, MTL::Buffer* zeroBuffer, int timestep) override;

private:
    int output_dim_;
    int inputSize_;
    int sequenceLength_;
    ReductionType reductionType_;
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif
