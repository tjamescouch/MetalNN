//
//  dropout-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once

#include "layer.h"
#import "data-source.h"
#include <Metal/Metal.hpp>

class DropoutLayer : public Layer {
public:
    DropoutLayer(float rate, int featureDim, int sequenceLength);
    ~DropoutLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;
    
    void setInputBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getOutputBufferAt(int timestep) const override;
    MTL::Buffer* getErrorBufferAt(int timestep) const override { return nullptr; };
    void setOutputErrorBufferAt(int timestep, MTL::Buffer* buffer) override;
    MTL::Buffer* getInputErrorBufferAt(int timestep) const override;
    void updateTargetBufferAt(DataSource&, int) override { }
    
    int outputSize() const override { return featureDim_; }
private:
    std::vector<MTL::Buffer*> bufferInputs_;
    std::vector<MTL::Buffer*> bufferOutputs_;
    std::vector<MTL::Buffer*> bufferInputErrors_;
    std::vector<MTL::Buffer*> bufferOutputErrors_;
    float rate_;
    int sequenceLength_;
    int featureDim_;
};
