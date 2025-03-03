//
//  function-dataset.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#pragma once

#include "dataset.h"
#include <functional>

class FunctionDataset : public Dataset {
public:
    using InputFunction = std::function<float(int, double)>;
    using TargetFunction = std::function<float(int, double)>;

    FunctionDataset(InputFunction inputFunc,
                    TargetFunction targetFunc,
                    int inputDim,
                    int outputDim);

    int inputDim() const override;
    int outputDim() const override;
    int numSamples() const override;

    void loadSample(int index, float* inputBuffer, float* targetBuffer);
    

    const std::vector<float>& inputAt(int index) override;
    const std::vector<float>& targetAt(int index) override;

private:
    InputFunction _inputFunc;
    TargetFunction _targetFunc;
    int _inputDim;
    int _outputDim;
};
