//
//  optimizable-layer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#pragma once
#include "common.h"

class OptimizableLayer {
public:
    OptimizableLayer();
    virtual ~OptimizableLayer();

    void buildAdamPipeline(MTL::Device* device, MTL::Library* library);
    void buildAdamBuffers(MTL::Device* device, size_t paramSize);
    void encodeAdamKernel(MTL::ComputeCommandEncoder* encoder,
                          MTL::Buffer* params,
                          MTL::Buffer* grads,
                          uint32_t paramCount);
    
    virtual MTL::Buffer* getParameterBuffer() const = 0;
    virtual MTL::Buffer* getGradientBuffer() const = 0;
    virtual uint32_t parameterCount() const = 0;

protected:
    MTL::Buffer* bufferM_;
    MTL::Buffer* bufferV_;
    uint32_t adamTimestep_;

private:
    MTL::ComputePipelineState* adamPipelineState_;

    // Adam hyperparameters (could later move to YAML config)
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
};
