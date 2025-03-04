//
//  layer-factory.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//

#include "layer-factory.h"
#include "dense-layer.h"
#include "dropout-layer.h"
#include "batch-normalization-layer.h"
#include "rnn-layer.h"

Layer* LayerFactory::createLayer(const LayerConfig& layerConfig,
                                 int input_dim,
                                 MTL::Device* device,
                                 MTL::Library* library) {
    Layer* layer = nullptr;
    int previousLayerOutputSize = input_dim;

    if (layerConfig.type == "Dense") {
        int outputSize = layerConfig.params.at("output_size").get_value<int>();
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        ActivationFunction activation = parseActivation(activationStr);
        layer = new DenseLayer(previousLayerOutputSize, outputSize, 1, activation);
        previousLayerOutputSize = outputSize;
    }
    else if (layerConfig.type == "Dropout") {
        float rate = layerConfig.params.at("rate").get_value<float>();
        layer = new DropoutLayer(rate, previousLayerOutputSize, 1);
    }
    else if (layerConfig.type == "BatchNormalization") {
        float epsilon = layerConfig.params.count("epsilon")
                        ? layerConfig.params.at("epsilon").get_value<float>()
                        : 0.001f;
        layer = new BatchNormalizationLayer(previousLayerOutputSize, 1, epsilon);
    }
    else if (layerConfig.type == "RNN") {
        int outputSize = layerConfig.params.at("output_size").get_value<int>();
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        ActivationFunction activation = parseActivation(activationStr);
        int timeSteps = layerConfig.time_steps;
        layer = new RNNLayer(previousLayerOutputSize, outputSize, timeSteps, activation);
        previousLayerOutputSize = outputSize;
    }
    else {
        throw std::invalid_argument("Unsupported layer type");
    }

    layer->buildPipeline(device, library);
    layer->buildBuffers(device);
    
    return layer;
}
