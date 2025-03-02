// input-layer.h
#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include "data-source.h"
#include <vector>

// Forward declarations for Metal classes.
namespace MTL {
    class Device;
    class Buffer;
}
class Layer;

class InputLayer : public Layer {
public:
    InputLayer(int inputDim, int sequenceLength);
    ~InputLayer();
    
    void buildBuffers(MTL::Device* device) override;
    void updateBufferAt(DataSource& ds, int timestep);
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {};
    void forward(MTL::CommandBuffer* cmdBuf) override {};
    void backward(MTL::CommandBuffer* cmdBuf) override {};

    void updateTargetBufferAt(DataSource& targetData, int timestep) const {};
    int outputSize() const override { return inputDim_; }
    void updateTargetBufferAt(DataSource& targetData, int timestep) override {};

    void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const override;

    void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const override;
    
    void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                             MTL::Buffer* zeroBuffer, int timestep) override {};
    
    
    int getParameterCount() const override;
    float getParameterAt(int index) const override;
    void setParameterAt(int index, float value) override;
    float getGradientAt(int index) const override;
    
    void onForwardComplete() override {};
    void onBackwardComplete() override {};
    
    int getSequenceLength();
    
    void debugLog() override {
#ifdef DEBUG_INPUT_LAYER
        for (int t = 0; t < sequenceLength_; t++) {
            float* outputs = static_cast<float*>(outputBuffers_[BufferType::Output][t]->contents());
            printf("[InputLayer DebugLog] outputs at timestep %d: %f, %f, %f\n",
                   t, outputs[0], outputs[1], outputs[2]);
        }
#endif
    }
    
private:
    int inputDim_;
    int sequenceLength_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> inputBuffers_;
    std::unordered_map<BufferType, std::vector<MTL::Buffer*>> outputBuffers_;
};

#endif // INPUT_LAYER_H
