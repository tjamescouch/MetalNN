#pragma once

#include "dataset.h"
#include <vector>
#include <string>

class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& imagesFilename, const std::string& labelsFilename);
    ~MNISTDataset() override = default;

    // Overrides from Dataset interface
    void loadData() override;
    
    float* getInputDataBuffer() override;
    float* getTargetDataBuffer() override;

    float* getInputDataAt(int timestep) override;
    float* getTargetDataAt(int timestep) override;
    float calculateLoss(const float* predictedData, int outputDim) override;

    int getDatasetSize() const override;

    // Existing specific methods
    int numSamples() const;
    int inputDim() const override;
    int outputDim() const override;

    const std::vector<float>& inputAt(int index);
    const std::vector<float>& targetAt(int index);
    
    void loadSample(int sampleIndex) override;

private:
    void loadImages(const std::string& imagesPath);
    void loadLabels(const std::string& labelsPath);

    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> targets_;


    int num_samples_;

    std::vector<float> currentInputBuffer_;
    std::vector<float> currentTargetBuffer_;
};
