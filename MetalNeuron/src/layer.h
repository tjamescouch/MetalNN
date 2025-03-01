//
//  layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef LAYER_H
#define LAYER_H

#include <functional>

#import "data-source.h"

// Forward declarations for Metal types.
namespace MTL {
class Buffer;
class Device;
class Library;
class CommandBuffer;
}

enum class ActivationFunction {
    Linear = 0,
    ReLU,
    Tanh,
    Sigmoid
};


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
    virtual void setInputBufferAt(int timestep, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getOutputBufferAt(int timestep) const = 0;
    virtual int outputSize() const = 0;
    virtual MTL::Buffer* getErrorBufferAt(int timestep) const = 0;
    virtual void updateTargetBufferAt(DataSource& targetData, int timestep) = 0;
    virtual void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getInputErrorBufferAt(int timestep) const = 0;
};

#endif // LAYER_H
