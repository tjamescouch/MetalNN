//
//  layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef LAYER_H
#define LAYER_H

#include <functional>

// Forward declarations for Metal types.
namespace MTL {
    class Device;
    class Library;
    class CommandBuffer;
}

class Layer {
public:
    virtual ~Layer() {}
    // Called to build pipeline states from the given device and library.
    virtual void buildPipeline(MTL::Device* device, MTL::Library* library) = 0;
    // Called to allocate any buffers needed for the layer.
    virtual void buildBuffers(MTL::Device* device) = 0;
    // Record commands for the forward pass.
    virtual void forward(MTL::CommandBuffer* cmdBuf) = 0;
    // Record commands for the backward pass.
    virtual void backward(MTL::CommandBuffer* cmdBuf) = 0;
};

#endif // LAYER_H
