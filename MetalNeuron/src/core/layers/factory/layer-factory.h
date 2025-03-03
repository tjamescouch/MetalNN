//
//  layer-factory.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#ifndef LAYER_FACTORY_H
#define LAYER_FACTORY_H

#include "layer.h"
#include "model-config.h"
#include <Metal/Metal.hpp>

class LayerFactory {
public:
    static Layer* createLayer(const LayerConfig& layerConfig,
                              int input_dim,
                              MTL::Device* device,
                              MTL::Library* library);
};

#endif // LAYER_FACTORY_H
