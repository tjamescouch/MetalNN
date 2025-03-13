//
//  sequence-to-vector-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#ifndef SEQUENCE_TO_VECTOR_LAYER_H
#define SEQUENCE_TO_VECTOR_LAYER_H


#include "layer.h"
#include <Metal/Metal.hpp>


class SequenceToVectorLayer : public Layer {
public:
    SequenceToVectorLayer(int inputDim, int featureDim, int batchSize, int sequenceLength);
    ~SequenceToVectorLayer() override;

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;

    void updateTargetBufferAt(const float*, int) override {}
    void updateTargetBufferAt(const float*, int, int) override {}
    
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    
    int inputSize() const override { return inputDim_; }
    int outputSize() const override { return featureDim_; }
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    int getSequenceLength() override { return sequenceLength_; };
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
    void debugLog() override {}
    
private:
    int sequenceLength_;
    int inputDim_;
    int featureDim_;
    bool isTerminal_;
    int batchSize_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    MTL::Device* _pDevice;
};

#endif
