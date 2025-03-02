//
//  layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include "data-source.h"

// Forward declarations for Metal types.
namespace MTL {
class Buffer;
class Device;
class Library;
class CommandBuffer;
}

class InputLayer;

enum class ActivationFunction {
    Linear = 0,
    ReLU,
    Tanh,
    Sigmoid
};

enum class BufferType : unsigned int {
    Input = 0,
    HiddenState,
    PrevHiddenState,
    Output,
    Debug,
    Targets,
    Gradients,
    Activation,
    OutputErrors,
    InputErrors,
    Delta
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
    
    virtual void setInputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getOutputBufferAt(BufferType type, int timestep) const = 0;
    virtual void setOutputBufferAt(BufferType type, int timestep, MTL::Buffer* buffer) = 0;
    virtual MTL::Buffer* getInputBufferAt(BufferType type, int timestep) const = 0;
    
    virtual int outputSize() const = 0;
    virtual void updateTargetBufferAt(DataSource& targetData, int timestep) = 0;
    virtual void connectInputBuffers(const Layer* previousLayer, const InputLayer* inputLayer,
                                     MTL::Buffer* zeroBuffer, int timestep) = 0;
    
    virtual void debugLog() = 0;
};

#endif // LAYER_H
