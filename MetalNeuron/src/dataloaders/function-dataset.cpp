//
//  function-dataset.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include "function-dataset.h"
#include <random>
#include "math-lib.h"

FunctionDataset::FunctionDataset(InputFunction inputFunc,
                                 TargetFunction targetFunc,
                                 int inputDim,
                                 int outputDim)
: _inputFunc(inputFunc)
, _targetFunc(targetFunc)
, _inputDim(inputDim)
, _outputDim(outputDim)
{ }

int FunctionDataset::inputDim() const {
    return _inputDim;
}

int FunctionDataset::outputDim() const {
    return _outputDim;
}

int FunctionDataset::numSamples() const {
    // Typically infinite or arbitrary for generated functions
    return 10000; // Placeholder number, adjust as needed
}

void FunctionDataset::loadSample(int index, float* inputBuffer, float* targetBuffer) {
    // Generate data at a random or sequential timestamp
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 2.0*M_PI);
    double effectiveTime = distribution(generator);

    for (int i = 0; i < _inputDim; ++i) {
        inputBuffer[i] = _inputFunc(i, effectiveTime);
    }
    
    for (int i = 0; i < _outputDim; ++i) {
        targetBuffer[i] = _targetFunc(i, effectiveTime);
    }
}

const std::vector<float>& FunctionDataset::inputAt(int index) {
    static std::vector<float> inputBuffer(_inputDim);
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < _inputDim; ++i) {
        inputBuffer[i] = _inputFunc(i, index);
    }
    return inputBuffer;
}

const std::vector<float>& FunctionDataset::targetAt(int index) {
    static std::vector<float> targetBuffer(_outputDim);
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < _outputDim; ++i) {
        targetBuffer[i] = _targetFunc(i, index);
    }
    return targetBuffer;
}
