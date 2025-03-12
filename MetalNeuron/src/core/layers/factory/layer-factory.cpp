//
//  layer-factory.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-04.
//
#include <iostream>
#include "layer-factory.h"
#include "dense-layer.h"
#include "dropout-layer.h"
#include "batch-normalization-layer.h"
#include "rnn-layer.h"
#include "map-reduce-layer.h"
#include "configuration-manager.h"

Layer* LayerFactory::createLayer(LayerConfig& layerConfig,
                                 MTL::Device* device,
                                 MTL::Library* library,
                                 bool isTerminal) {
    
    std::cout << "Getting global parameters..." << std::endl;
    auto config = ConfigurationManager::instance().getConfig();
    auto globaLearningRate = config->training.optimizer.learning_rate;
    auto batchSize = config->training.batch_size;
    std::cout << "Getting common layer parameters..." << std::endl;
    int inputSize = layerConfig.params.at("input_size").get_value<int>();
    int outputSize = layerConfig.params.at("output_size").get_value<int>();
    
    auto learningRate = layerConfig.params["learning_rate"].get_value_or<float>(globaLearningRate);
    learningRate = learningRate > 0 ? learningRate : globaLearningRate;
    
    Layer* layer = nullptr;

    if (layerConfig.type == "Dense") {
        std::cout << "Creating dense layer..." << std::endl;
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");

        
        ActivationFunction activation = parseActivation(activationStr);
        layer = (new DenseLayer(inputSize, outputSize, 1, activation, batchSize))
                    ->setLearningRate(learningRate)
                    ->setInitializer(initializer);
        
    }
    else if (layerConfig.type == "Dropout") {
        std::cout << "Creating dropout layer..." << std::endl;
        float rate = layerConfig.params.at("rate").get_value_or<float>(0.3);
        layer = new DropoutLayer(rate, inputSize, outputSize, batchSize, 1);
    }
    else if (layerConfig.type == "BatchNormalization") {
        std::cout << "Creating batch normalization layer..." << std::endl;
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
        layer = new BatchNormalizationLayer(inputSize, outputSize, batchSize, 1, learningRate, epsilon);
    }
    else if (layerConfig.type == "RNN") {
        std::cout << "Creating RNN layer..." << std::endl;
        int outputSize = layerConfig.params.at("output_size").get_value<int>();
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        ActivationFunction activation = parseActivation(activationStr);
        int timeSteps = layerConfig.time_steps;
        
        
        auto learningRate = layerConfig.params["learning_rate"].get_value_or<float>(globaLearningRate);
        learningRate = learningRate > 0 ? learningRate : globaLearningRate;
        
        layer = new RNNLayer(inputSize, outputSize, timeSteps, activation, learningRate);
    }
    else if (layerConfig.type == "MapReduce") {
        std::cout << "Creating MapReduce layer..." << std::endl;
        auto reductionType = layerConfig.params.at("reduction_type").get_value<std::string>();
        layer = new MapReduceLayer(inputSize, outputSize, parseReductionType(reductionType));
    }
    else {
        throw std::invalid_argument("Unsupported layer type");
    }
    layer->setIsTerminal(isTerminal);
    layer->buildPipeline(device, library);
    layer->buildBuffers(device);
    
    return layer;
}
