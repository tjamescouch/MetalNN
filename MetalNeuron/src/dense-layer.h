#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "input-layer.h"
#include "data-source.h"
#include <Metal/Metal.hpp>
#include <vector>



class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim, int sequenceLength, ActivationFunction activationFunction);
    ~DenseLayer();
    
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void updateTargetBufferAt(DataSource& targetData, int timestep) override;
    
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;
    

    int outputSize() const override;
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const override;
    
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    
    void debugLog() override {
        
        float* weights = static_cast<float*>(bufferWeights_->contents());
        printf("[DenseLayer DebugLog] Weights sample: %f, %f, %f\n", weights[0], weights[1], weights[2]);
        
        // Optionally log biases or other important states:
        float* biases = static_cast<float*>(bufferBias_->contents());
        printf("[DenseLayer DebugLog] Biases sample: %f, %f, %f\n", biases[0], biases[1], biases[2]);
    }
    
private:
    int inputDim_;
    int outputDim_;
    int sequenceLength_;
    
    ActivationFunction activation_;
    
    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
};

#endif // DENSE_LAYER_H
