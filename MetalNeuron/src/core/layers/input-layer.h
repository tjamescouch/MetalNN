// input-layer.h
#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include <vector>

// Forward declarations for Metal classes.
namespace MTL {
    class Device;
    class Buffer;
}
class Layer;

class InputLayer : public Layer {
public:
    InputLayer(int inputDim, int sequenceLength, int batchSize);
    ~InputLayer();
    
    void buildBuffers(MTL::Device* device) override;
    void updateBufferAt(const float*, int timestep);
    void updateBufferAt(const float*, int timestep, int batchSize);
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {};
    void forward(MTL::CommandBuffer* cmdBuf, int batchSize) override {};
    void backward(MTL::CommandBuffer* cmdBuf, int batchSize) override {};

    int outputSize() const override { return inputDim_; }
    void updateTargetBufferAt(const float* targetData, int timestep) override {};
    void updateTargetBufferAt(const float* targetData, int timestep, int batchSize) override {};

    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) override;
    
    void connectForwardConnections(Layer* previousLayer, Layer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override {};
    void connectBackwardConnections(Layer* previousLayer, Layer* inputLayer,
                                    MTL::Buffer* zeroBuffer, int timestep) override {};
    
    
    void onForwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {};
    void onBackwardComplete(MTL::CommandQueue* _pCommandQueue, int batchSize) override {};
    
    int getSequenceLength() override;
    
    void saveParameters(std::ostream& os) const override;
    void loadParameters(std::istream& is) override;
    
    void setIsTerminal(bool isTerminal) override { isTerminal_ = isTerminal; };
    
    void debugLog() override {
#ifdef DEBUG_INPUT_LAYER
        for (int t = 0; t < sequenceLength_; t++) {
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[InputLayer Output Debug] timestep %d: ", t);
            for(int i = 0; i < outputBuffers_[BufferType::Output][t]->length()/sizeof(float); ++i)
                printf(" %f, ", outputs[i]);
            printf("\n");
        }
#endif
    }
    
private:
    int inputDim_;
    int sequenceLength_;
    bool isTerminal_;
    int batchSize_;
    
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
};

#endif // INPUT_LAYER_H
