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
#include "map-reduce-layer.h"
#include "configuration-manager.h"

Layer* LayerFactory::createLayer(LayerConfig& layerConfig,
                                 int input_dim,
                                 MTL::Device* device,
                                 MTL::Library* library,
                                 bool isTerminal) {
    auto config = ConfigurationManager::instance().getConfig();
    auto globaLearningRate = config->training.optimizer.learning_rate;
    auto batchSize = config->training.batch_size;
    
    Layer* layer = nullptr;
    int previousLayerOutputSize = input_dim;

    if (layerConfig.type == "Dense") {
        int outputSize = layerConfig.params.at("output_size").get_value<int>();
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
        
        auto learningRate = layerConfig.params["learning_rate"].get_value_or<float>(globaLearningRate);
        learningRate = learningRate > 0 ? learningRate : globaLearningRate;
        
        ActivationFunction activation = parseActivation(activationStr);
        layer = (new DenseLayer(previousLayerOutputSize, outputSize, 1, activation, batchSize))
                    ->setLearningRate(learningRate)
                    ->setInitializer(initializer);
        
        previousLayerOutputSize = outputSize;
    }
    else if (layerConfig.type == "Dropout") {
        float rate = layerConfig.params.at("rate").get_value<float>();
        layer = new DropoutLayer(rate, previousLayerOutputSize, 1);
    }
    else if (layerConfig.type == "BatchNormalization") {
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
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
    else if (layerConfig.type == "MapReduce") {
        auto reductionType = layerConfig.params.at("reduction_type").get_value<std::string>();
        layer = new MapReduceLayer(previousLayerOutputSize, parseReductionType(reductionType));
        previousLayerOutputSize = 1;
    }
    else {
        throw std::invalid_argument("Unsupported layer type");
    }
    layer->setIsTerminal(isTerminal);
    layer->buildPipeline(device, library);
    layer->buildBuffers(device);
    
    return layer;
}
