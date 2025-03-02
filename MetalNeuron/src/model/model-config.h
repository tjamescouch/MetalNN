//
//  model-config.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-28.
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fkYAML/node.hpp>

// Layer configuration
struct LayerConfig {
    std::string type;
    std::map<std::string, fkyaml::node> params;
    int time_steps = -1;
};

// Optimizer configuration
struct OptimizerConfig {
    std::string type;
    float learning_rate;
    std::map<std::string, fkyaml::node> parameters;
};

// Training configuration
struct TrainingConfig {
    OptimizerConfig optimizer;
    int epochs;
    int batch_size;
};

struct ModelDataSet {
    std::string type;
    std::string images;
    std::string labels;
};


// Overall model configuration
class ModelConfig {
public:
    int first_layer_time_steps = -1;
    std::string name;
    std::vector<LayerConfig> layers;
    TrainingConfig training;
    std::map<std::string, fkyaml::node> metadata;
    ModelDataSet dataset;
    
    static ModelConfig loadFromFile(const std::string& filePath);
};
