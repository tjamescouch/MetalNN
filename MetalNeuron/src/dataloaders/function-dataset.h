#pragma once

#include "dataset.h"
#include <functional>
#include <vector>
#include <numeric> 
#include <algorithm>
#include <random>

using InputFunction = std::function<float(int, double)>;
using TargetFunction = std::function<float(int, double)>;

class FunctionDataset : public Dataset {
public:
    FunctionDataset(InputFunction inputFunc, TargetFunction targetFunc,
                                     int inputDim, int outputDim, int datasetSize);
    ~FunctionDataset() override = default;

    void loadData() override;
    
    float* getInputDataBuffer() override;
    float* getTargetDataBuffer() override;

    float* getInputDataAt(int timestep) override;
    float* getTargetDataAt(int timestep) override;
    float calculateLoss(const float* predictedData, int outputDim) override;

    int getDatasetSize() const override;
    
    int inputDim() const override { return inputDim_; };
    int outputDim() const override { return outputDim_; };
    
    void loadSample(int sampleIndex) override;
    

    int numSamples() const override;
    

private:
    InputFunction inputFunc_;
    TargetFunction targetFunc_;
    int inputDim_;
    int outputDim_;
    int datasetSize_;
    double offset_ = 0;

    std::vector<int> shuffledIndices_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> targets_;
    
    std::vector<float> currentInputBuffer_;
    std::vector<float> currentTargetBuffer_;
    
    void shuffleIndices();
    void generateDataset(double offset);
};
