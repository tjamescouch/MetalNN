//
//  model-config.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-28.
//
#include "model-config.h"
#include <fkYAML/node.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

ModelConfig ModelConfig::loadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file) {
        throw std::runtime_error("Could not open YAML file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    auto config = fkyaml::node::deserialize(buffer.str());
    ModelConfig modelConfig;

    // Load basic fields
    modelConfig.name = config["name"].get_value<std::string>();

    // Load layers
    for (const auto& layer : config["layers"]) {
        LayerConfig layerConfig;
        layerConfig.type = layer["type"].get_value<std::string>();
        layerConfig.time_steps = layer["time_steps"].get_value<int>();
        
        if (modelConfig.first_layer_time_steps == -1) {
            modelConfig.first_layer_time_steps = layer["time_steps"].get_value<int>();
        }

        // Load layer parameters
        for (const auto& param : layer.as_map()) {
            const std::string& key = param.first.get_value<std::string>();
            if (key != "type") {
                layerConfig.params[key] = param.second;
            }
        }

        modelConfig.layers.push_back(layerConfig);
    }

    // Load optimizer
    const auto optimizerNode = config["training"]["optimizer"];
    OptimizerConfig optimizerConfig;
    optimizerConfig.type = optimizerNode["type"].get_value<std::string>();
    optimizerConfig.learning_rate = optimizerNode["learning_rate"].get_value<float>();

    if (optimizerNode.contains("parameters")) {
        for (const auto& param : optimizerNode["parameters"].as_map()) {
            optimizerConfig.parameters[param.first.get_value<std::string>()] = param.second;
        }
    }

    // Load training details
    modelConfig.training.optimizer = optimizerConfig;
    modelConfig.training.epochs = config["training"]["epochs"].get_value<int>();
    modelConfig.training.batch_size = config["training"]["batch_size"].get_value<int>();

    // Load metadata (optional)
    if (config.contains("metadata")) {
        for (const auto& item : config["metadata"].as_map()) {
            modelConfig.metadata[item.first.get_value<std::string>()] = item.second;
        }
    }

    return modelConfig;
}
