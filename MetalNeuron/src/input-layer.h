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

    // Updates input buffer for a specific timestep.
    void updateBufferAt(DataSource& ds, int timestep);
    
    // Retrieves the buffer for a specific timestep.
    MTL::Buffer* getBufferAt(int timestep) const;
    
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {};
    void forward(MTL::CommandBuffer* cmdBuf) override {};
    void backward(MTL::CommandBuffer* cmdBuf) override {};
    
private:
    int inputDim_;
    int sequenceLength_;
    std::vector<MTL::Buffer*> bufferInputs_;
};

#endif // INPUT_LAYER_H
