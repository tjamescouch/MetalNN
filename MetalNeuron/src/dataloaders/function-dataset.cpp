#include <random>
#include <stdexcept>

#include "math-lib.h"
#include "training-manager.h"
#include "function-dataset.h"



std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0, 2*M_PI);


FunctionDataset::FunctionDataset(InputFunction inputFunc, TargetFunction targetFunc,
                                 int inputDim, int outputDim, int datasetSize)
: inputFunc_(inputFunc),
  targetFunc_(targetFunc),
  inputDim_(inputDim),
  outputDim_(outputDim),
  datasetSize_(datasetSize),
  currentInputBuffer_(inputDim, 0.0f),
  currentTargetBuffer_(outputDim, 0.0f),
  inputs_(datasetSize, std::vector<float>(inputDim)),
  targets_(datasetSize, std::vector<float>(outputDim)) {
}



float* FunctionDataset::getInputDataAt(int timestep) {
    if (timestep < 0 || timestep >= datasetSize_) {
        throw std::out_of_range("❌ getInputDataAt: timestep out of range");
    }
    return inputs_[timestep].data();
}

float* FunctionDataset::getTargetDataAt(int timestep) {
    if (timestep < 0 || timestep >= datasetSize_) {
        throw std::out_of_range("❌ getTargetDataAt: timestep out of range");
    }
    return targets_[timestep].data();
}

int FunctionDataset::getDatasetSize() const {
    return datasetSize_;
}

float FunctionDataset::calculateLoss(const float* predictedData, int outputDim, const float* targetData) {
    float mse = 0.0f;

    for (int i = 0; i < outputDim; ++i) {
        float diff = predictedData[i] - targetData[i];
        mse += diff * diff;
    }

    mse /= static_cast<float>(outputDim);
    return mse;
}



void FunctionDataset::loadData() {
    bool isTraining = TrainingManager::instance().isTraining();
    if (isTraining) {
        shuffleIndices();
        generateDataset(offset_);
    }else {
        generateDataset(offset_++);
    }
}

void FunctionDataset::generateDataset(double offset) {
    inputs_.resize(datasetSize_);
    targets_.resize(datasetSize_);
    for (int t = 0; t < datasetSize_; ++t) {
        double effectiveTime = static_cast<double>(t);
        for (int i = 0; i < inputDim_; ++i) {
            inputs_[t][i] = inputFunc_(i, effectiveTime + offset);
        }
        for (int i = 0; i < outputDim_; ++i) {
            targets_[t][i] = targetFunc_(i, effectiveTime + offset);
        }
    }
}

void FunctionDataset::loadNextSample() {
    loadData();
}

void FunctionDataset::shuffleIndices() {
    offset_ = (int)round(distribution(generator));
}

int FunctionDataset::numSamples() const {
    // Returns the total number of samples in this dataset
    return datasetSize_;
}

float* FunctionDataset::getInputDataBuffer() {
    return inputs_[0].data();
}

float* FunctionDataset::getTargetDataBuffer() {
    return targets_[0].data();
}


float* FunctionDataset::getInputDataAt(int timestep, int batchIndex) {
    int index = (offset_ + batchIndex) % inputs_.size();
    return inputs_[index].data();
}

float* FunctionDataset::getTargetDataAt(int timestep, int batchIndex) {
    int index = (offset_ + batchIndex) % targets_.size();
    return targets_[index].data();
}
