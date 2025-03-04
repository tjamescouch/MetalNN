#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"
#include "optimizer.h"
#include "input-layer.h"
#include "data-source.h"
#include <Metal/Metal.hpp>
#include <vector>



class DenseLayer : public Layer {
public:
    DenseLayer(int inputDim, int outputDim, int _unused, ActivationFunction activationFunction);
    ~DenseLayer();
    
    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    
    void updateTargetBufferAt(const float* targetData, int timestep) override;
    
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;
    

    int outputSize() const override;
    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    
    void connectInputBuffers(Layer* previousLayer, InputLayer* inputLayer,
                                         MTL::Buffer* zeroBuffer, int timestep) override;
    
    
    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;
    
    void onForwardComplete() override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue) override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void debugLog() override;
    
    int getSequenceLength() override { return sequenceLength_; };
    
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
private:
    int inputDim_;
    int outputDim_;
    int sequenceLength_;
    bool isTerminal_;
    
    ActivationFunction activation_;
    
    MTL::Buffer* bufferWeights_;
    MTL::Buffer* bufferBias_;
    MTL::Buffer* bufferDecay_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
    
    
    MTL::ComputePipelineState* forwardPipelineState_;
    MTL::ComputePipelineState* backwardPipelineState_;
    
    std::unique_ptr<Optimizer> optimizerWeights_;
    std::unique_ptr<Optimizer> optimizerBiases_;
};

#endif // DENSE_LAYER_H
