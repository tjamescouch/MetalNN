#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "optimizer.h"
#include "input-layer.h"
#include <Metal/Metal.hpp>
#include <vector>



class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activationFunction, int batchSize);
    ~DenseLayer();
    
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void updateTargetBufferAt(const float* targetData) override;
    void updateTargetBufferAt(const float* targetData, int batchSize) override;
    
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;

    void resetErrors() override;

    int inputSize() const override;
    int outputSize() const override;
    
    void setInputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBuffer(BufferType type) override;

    void setOutputBuffer(BufferType type, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBuffer(BufferType type) override;
    
    void connectForwardConnections(Layer* previousLayer) override;
    void connectBackwardConnections(Layer* previousLayer) override;
    
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void debugLog() override;
        
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; }
    DenseLayer* setLearningRate(float learningRate) { learningRate_ = learningRate; return this; }
    DenseLayer* setInitializer(std::string initializer) { initializer_ = initializer; return this; }
    
private:
    int inputDim_;
    int outputDim_;
    int sequenceLength_;
    bool isTerminal_;
    float learningRate_;
    int batchSize_;
    float decayRate_ = 1.0f;
    float decay_ = 1.0f;
    
    
    std::string initializer_;
    
    ActivationFunction activation_;
    
    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unique_ptr<Optimizer> optimizerWeights_;
    std::unique_ptr<Optimizer> optimizerBiases_;
};

#endif // DENSE_LAYER_H
