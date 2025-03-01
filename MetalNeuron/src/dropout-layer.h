//
//  dropout-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#pragma once

#include "layer.h"
#include <Metal/Metal.hpp>

class DropoutLayer : public Layer {
public:
    DropoutLayer(float rate, int sequenceLength);
    ~DropoutLayer();

    void buildPipeline(MTL::Device* device, MTL::Library* library) override;
    void buildBuffers(MTL::Device* device) override;
    void forward(MTL::CommandBuffer* cmdBuf) override;
    void backward(MTL::CommandBuffer* cmdBuf) override;

private:
    float rate_;
    int sequenceLength_;
    // Add Metal buffers and pipelines as needed
};
