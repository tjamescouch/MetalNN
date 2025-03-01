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

class InputLayer : public Layer {
public:
    InputLayer(int inputDim, int sequenceLength);
    virtual ~InputLayer();
    
    void buildBuffers(MTL::Device* device) override;
    void updateBufferAt(DataSource& ds, int timestep);
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {};
    void forward(MTL::CommandBuffer* cmdBuf) override {};
    void backward(MTL::CommandBuffer* cmdBuf) override {};
    void setInputBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(int timestep) const override;
    MTL::Buffer* getErrorBufferAt(int timestep) const override;
    void updateTargetBufferAt(DataSource& targetData, int timestep) const {};
    int outputSize() const override { return inputDim_; }
    void updateTargetBufferAt(DataSource& targetData, int timestep) override {};
    void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputErrorBufferAt(int timestep) const override;
    
private:
    int inputDim_;
    int sequenceLength_;
    std::vector<MTL::Buffer*> bufferInputs_;
};

#endif // INPUT_LAYER_H
