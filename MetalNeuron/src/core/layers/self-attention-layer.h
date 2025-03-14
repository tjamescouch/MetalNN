#pragma once
#include "layer.h"
#include "optimizer.h"
#include <vector>

class SelfAttentionLayer : public Layer {
public:
    SelfAttentionLayer(MTL::Device* device, int inputDim, int seqLength, int modelDim);
    ~SelfAttentionLayer();

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
    int outputSize() const override { return modelDim_; }
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void setIsTerminal(bool isTerminal) override;
    
    void debugLog() override;
    
   

private:
    void initializeWeights();
    
    MTL::Device* device_;
    int inputDim_;
    int seqLength_;
    int modelDim_;
    bool isTerminal_;

    // Buffers
    MTL::Buffer* bufferQ_;
    MTL::Buffer* bufferK_;
    MTL::Buffer* bufferV_;
    
    MTL::Buffer* weightsQ_;
    MTL::Buffer* weightsK_;
    MTL::Buffer* weightsV_;
    MTL::Buffer* outputProjection_;

    std::unique_ptr<Optimizer> optimizerWeightsQ_;
    std::unique_ptr<Optimizer> optimizerWeightsK_;
    std::unique_ptr<Optimizer> optimizerWeightsV_;
    std::unique_ptr<Optimizer> optimizerOutputProjection_;

    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unordered_map<BufferType, MTL::Buffer*> inputBuffers_;
    std::unordered_map<BufferType, MTL::Buffer*> outputBuffers_;
};
