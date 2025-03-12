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
    
    void updateTargetBufferAt(const float* targetData, int timestep) override;
    void updateTargetBufferAt(const float* targetData, int timestep, int batchSize) override;
    
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override;
    

    int inputSize() const override;
    int outputSize() const override;
    
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) override;
    
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void debugLog() override;
    
    int getSequenceLength() override { return sequenceLength_; };
    
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
    static int layerCounter;
    int layerIndex = -1;
    
    
    std::string initializer_;
    
    ActivationFunction activation_;
    
    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;
    MTL::Buffer* bufferLearningRate_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unique_ptr<Optimizer> optimizerWeights_;
    std::unique_ptr<Optimizer> optimizerBiases_;
};

#endif // DENSE_LAYER_H
