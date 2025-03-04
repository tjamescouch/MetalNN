//
//  gradient-checker.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#ifdef FUTURE_CLASS
#include "gradient-checker.h"
#include <cassert>
#include <iostream>
#include <cmath>

GradientChecker::GradientChecker(NeuralEngine* engine, DataSourceManager* dataSource)
    : engine_(engine), dataSource_(dataSource) {}

float GradientChecker::computeLoss() {
    engine_->computeForwardSync();  // synchronous forward pass
    
    float mse = 0.0f;
    Layer* outputLayer = engine_->dynamicLayers_.back();

    float* outputData = static_cast<float*>(
        outputLayer->getOutputBufferAt(BufferType::Output, 0)->contents()
    );

    float* targetData = dataSource_->y.get_data_buffer_at(0);
    int outputDim = outputLayer->outputSize();

    for (int i = 0; i < outputDim; ++i) {
        float diff = targetData[i] - outputData[i];
        mse += diff * diff;
    }

    return mse / outputDim;
}

float GradientChecker::numericalGradient(Layer* layer, int paramIndex, float epsilon) {
    float originalValue = layer->getParameterAt(paramIndex);

    // Perturb positively
    layer->setParameterAt(paramIndex, originalValue + epsilon);
    float lossPlus = computeLoss();

    // Perturb negatively
    layer->setParameterAt(paramIndex, originalValue - epsilon);
    float lossMinus = computeLoss();

    // Reset parameter
    layer->setParameterAt(paramIndex, originalValue);

    return (lossPlus - lossMinus) / (2.0f * epsilon);
}

void GradientChecker::checkLayerGradients(Layer* layer, float tolerance) {
    engine_->computeForwardSync();
    engine_->computeBackwardSync();

    int paramCount = layer->getParameterCount();

    for (int i = 0; i < paramCount; ++i) {
        float analyticalGrad = layer->getGradientAt(i);
        float numericalGrad = numericalGradient(layer, i);

        float diff = fabs(numericalGrad - analyticalGrad);
        if (diff > tolerance) {
            std::cout << "[Gradient Check FAILED] Param #" << i
                      << " Analytical: " << analyticalGrad
                      << ", Numerical: " << numericalGrad
                      << ", Diff: " << diff << "\n";
        } else {
            std::cout << "[Gradient Check Passed] Param #" << i
                      << " Diff: " << diff << "\n";
        }
    }
}
#endif
