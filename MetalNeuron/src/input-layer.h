//
//  input-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include "data-source.h"  // Assumes DataSource provides getData() and getSize()

// Forward declarations for Metal classes.
namespace MTL {
class Device;
class Buffer;
}

class InputLayer : public Layer {
public:
    // Constructor takes the number of input elements.
    InputLayer(int inputDim);
    virtual ~InputLayer();
    
    // Allocate the Metal buffer for the input.
    void buildBuffers(MTL::Device* device) override;
    
    // Update the Metal buffer from the provided DataSource.
    // Assumes ds.getData() returns a pointer to the data and ds.getSize() returns the size in bytes.
    void updateBuffer(DataSource& ds);
    
    // Returns the Metal buffer holding the input data.
    MTL::Buffer* getBuffer() const;
    
    // Called to build pipeline states from the given device and library.
    void buildPipeline(MTL::Device* device, MTL::Library* library) override {
        
    };
    
    // Record commands for the forward pass.
    void forward(MTL::CommandBuffer* cmdBuf) override {
        
    };
    // Record commands for the backward pass.
    void backward(MTL::CommandBuffer* cmdBuf) override {
        
    };
    
private:
    int inputDim_;
    MTL::Buffer* bufferInput_;
};

#endif // INPUT_LAYER_H
