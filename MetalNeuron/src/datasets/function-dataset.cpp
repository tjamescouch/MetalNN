#include <random>
#include <stdexcept>

#include "math-lib.h"
#include "training-manager.h"
#include "function-dataset.h"



std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0, 200*M_PI);


FunctionDataset::FunctionDataset(InputFunction inputFunc, TargetFunction targetFunc,
                                 int inputDim, int outputDim, int datasetSize)
: inputFunc_(inputFunc),
  targetFunc_(targetFunc),
  inputDim_(inputDim),
  outputDim_(outputDim),
  datasetSize_(datasetSize),
  currentInputBuffer_(inputDim, 0.0f),
  currentTargetBuffer_(outputDim, 0.0f),
  inputs_(0),
  targets_(0) {
}

FunctionDataset::~FunctionDataset() {
    
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



void FunctionDataset::loadData(int batchSize) {
    bool isTraining = TrainingManager::instance().isTraining();
    if (isTraining) {
        shuffleIndices();
        generateBatch(offset_, batchSize);
    }else {
        generateBatch(offset_++, batchSize);
    }
}

void FunctionDataset::generateBatch(double offset, int batchSize) {
    const int batchDataSize = inputDim_ * batchSize;
    inputs_.resize(batchDataSize);
    targets_.resize(batchDataSize);
    for (int t = 0; t < batchSize; ++t) {
        for (int i = 0; i < inputDim_; ++i) {
            int index = t * inputDim_ + i;
            inputs_[index] = inputFunc_(index, offset + offset_);
            targets_[index] = targetFunc_(index, offset + offset_);
        }
    }
}

void FunctionDataset::loadNextBatch(int batchSize) {
    loadData(batchSize);
}

void FunctionDataset::shuffleIndices() {
    offset_ = (int)round(distribution(generator));
}

int FunctionDataset::numSamples() const {
    // Returns the total number of samples in this dataset
    return datasetSize_;
}


float* FunctionDataset::getInputDataAt(int timestep, int _batchIndex) {
    return inputs_.data();
}

float* FunctionDataset::getTargetDataAt(int timestep, int _batchIndex) {
    return targets_.data();
}
