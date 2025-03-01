//
//  weight-initializer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-01.
//

#include "weight-initializer.h"

void WeightInitializer::initializeXavier(float* buffer, int inputDim, int outputDim) {
    float xavier_scale = sqrtf(6.0f / (inputDim + outputDim));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-xavier_scale, xavier_scale);
    for (int i = 0; i < inputDim * outputDim; ++i)
        buffer[i] = dist(rng);
}

void WeightInitializer::initializeBias(float* buffer, int dim, float scale) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (int i = 0; i < dim; ++i)
        buffer[i] = dist(rng);
}
