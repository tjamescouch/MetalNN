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
#include "multi-head-attention-layer.h"
#include "batch-normalization-layer.h"
#include "layer-normalization-layer.h"
#include "residual-connection-layer.h"
#include "self-attention-layer.h"
#include "map-reduce-layer.h"
#include "configuration-manager.h"

const char* inputLayerName = "input";

LayerFactory::LayerFactory(Layer* _pInputLayer) {
    _pInputLayer->setName(inputLayerName);
    layerMap_[inputLayerName] = _pInputLayer;
}

Layer* LayerFactory::createLayer(LayerConfig& layerConfig,
                                 MTL::Device* device,
                                 MTL::Library* library,
                                 bool isTerminal) {

    std::cout << "Getting global parameters..." << std::endl;
    auto config = ConfigurationManager::instance().getConfig();
    auto batchSize = config->training.batch_size;

    std::cout << "Getting common layer parameters..." << std::endl;
    int inputSize = 0;
    int outputSize = 0;
    int sequenceLength = 1;  // default for non-sequence layers

    // Explicitly handle shapes for sequence-aware layers
    if (layerConfig.params.contains("input_shape")) {
        int inputShape[2] = {};
        layerConfig.params["input_shape"].get_value_inplace(inputShape);
        sequenceLength = inputShape[0];
        inputSize = inputShape[1];
    } else {
        inputSize = layerConfig.params.at("input_size").get_value<int>();
    }

    if (layerConfig.params.contains("output_shape")) {
        int outputShape[2] = {};
        layerConfig.params["output_shape"].get_value_inplace(outputShape);
        outputSize = outputShape[1];
    } else {
        outputSize = layerConfig.params.at("output_size").get_value<int>();
    }

    // Provide a default sequential numeric ID if name not explicitly provided
    std::string layerName = layerConfig.params["name"].get_value_or<std::string>(
        "layer_" + std::to_string(layerIdCounter_++)
    );
    layerConfig.params["name"] = layerName;

    auto learningRate = layerConfig.learning_rate;

    Layer* layer = nullptr;

    if (layerConfig.type == "Dense") {
        std::cout << "Creating dense layer..." << std::endl;
        auto activationStr = layerConfig.params.at("activation").get_value<std::string>();
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");

        ActivationFunction activation = parseActivation(activationStr);
        layer = (new DenseLayer(inputSize, outputSize, 1, activation, batchSize))
                    ->setLearningRate(learningRate)
                    ->setInitializer(initializer);
        
    } else if (layerConfig.type == "Dropout") {
        std::cout << "Creating dropout layer..." << std::endl;
        float rate = layerConfig.params.at("rate").get_value_or<float>(0.3);
        layer = new DropoutLayer(rate, inputSize, outputSize, batchSize, 1);
        
    } else if (layerConfig.type == "SelfAttention") {
        std::cout << "Creating self attention layer..." << std::endl;
        
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
        
        layer = (new SelfAttentionLayer(inputSize, outputSize, sequenceLength, batchSize))->setInitializer(initializer);
        
    } else if (layerConfig.type == "MultiHeadAttention") {
        std::cout << "Creating multi-head attention layer..." << std::endl;
        int num_heads = layerConfig.params.at("num_heads").get_value_or<int>(2);
        auto initializer = layerConfig.params["initializer"].get_value_or<std::string>("xavier");
        
        layer = (new MultiHeadAttentionLayer(inputSize, outputSize, sequenceLength, batchSize, num_heads))->setInitializer(initializer);
        
    } else if (layerConfig.type == "BatchNormalization") {
        std::cout << "Creating batch normalization layer..." << std::endl;
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
        layer = new BatchNormalizationLayer(inputSize, outputSize, batchSize, 1, learningRate, epsilon);
        
    } else if (layerConfig.type == "LayerNormalization") {
        std::cout << "Creating layer normalization layer..." << std::endl;
        float epsilon = layerConfig.params["epsilon"].get_value_or<float>(1e-5f);
        epsilon = epsilon > 0 ? epsilon : 1e-5f;
        layer = new LayerNormalizationLayer(inputSize, sequenceLength, batchSize, learningRate, epsilon);
        
    } else if (layerConfig.type == "ResidualConnection") {
        auto from = layerConfig.params.at("from_layer").get_value<std::string>();
        std::cout << "Creating residual connection layer from " << from << "..." << std::endl;
        layer = (new ResidualConnectionLayer(inputSize, batchSize))
                    ->setResidualInput(layerMap_[from]->getOutputBufferAt(BufferType::Output));
        
    } else if (layerConfig.type == "MapReduce") {
        std::cout << "Creating MapReduce layer..." << std::endl;
        auto reductionType = layerConfig.params.at("reduction_type").get_value<std::string>();
        layer = new MapReduceLayer(inputSize, outputSize, parseReductionType(reductionType));
        
    } else {
        throw std::invalid_argument("Unsupported layer type");
    }

    layerMap_[layerName] = layer;
    layer->setIsTerminal(isTerminal);
    layer->setName(layerName);
    layer->buildPipeline(device, library);
    layer->buildBuffers(device);

    return layer;
}
