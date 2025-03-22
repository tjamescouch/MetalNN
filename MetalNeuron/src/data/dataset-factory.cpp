//
//  dataset-factory.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#include "dataset-factory.h"
#include "mnist-dataset.h"
#include "function-dataset.h"
#include "model-config.h"
#include <stdexcept>
#include "math-lib.h"

// Explicitly creates dataset based on model configuration
Dataset* DatasetFactory::createDataset(const ModelConfig* pConfig) {
    if (pConfig->dataset.type == "mnist") {
        return new MNISTDataset(
                                    pConfig->dataset.images,
                                    pConfig->dataset.labels
                                    );
    } else if (pConfig->dataset.type == "function") {
        int inputShape[2] = {};
        pConfig->layers.front().params.at("output_shape").get_value_inplace(inputShape);
        int inputSequenceLength = inputShape[0];
        int targetSequenceLength = 1;
        int featureDim = inputShape[1];
        int datasetSize = pConfig->dataset.dataset_size;

        int outputDim = 0;
        if (pConfig->layers.back().params.contains("output_shape")) {
            int outputShape[2] = {};
            pConfig->layers.back().params.at("output_shape").get_value_inplace(outputShape);
            targetSequenceLength = outputShape[0];
            outputDim = outputShape[1];
        } else {
            outputDim = pConfig->layers.back().params.at("output_size").get_value<int>();
        }

        return new FunctionDataset(mathlib::inputFunc, mathlib::targetFunc,
                                       inputSequenceLength,
                                       targetSequenceLength,
                                       featureDim,
                                       outputDim,
                                       datasetSize);
    } else {
        throw std::runtime_error("Unsupported dataset type: " + pConfig->dataset.type);
    }
}
