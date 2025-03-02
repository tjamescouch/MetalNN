// DataSourceManager.cpp
#include "data-source-manager.h"
#include <stdexcept>

// Existing constructor preserved unchanged for backward compatibility
DataSourceManager::DataSourceManager(int inputDim, int hiddenDim, int outputDim, int sequenceLength)
    : inputDim_(inputDim),
      hiddenDim_(hiddenDim),
      outputDim_(outputDim),
      sequenceLength_(sequenceLength),
      dataset_(nullptr),
      ownsDataset_(false),
      x(inputDim, 1, sequenceLength),
      h(hiddenDim, 1, sequenceLength),
      y(outputDim, 1, sequenceLength) {
    allocateBuffers();
}

// New generalized constructor
DataSourceManager::DataSourceManager(Dataset* dataset, int sequenceLength)
    : dataset_(dataset),
      ownsDataset_(false), // Adjust if DataSourceManager owns the dataset lifetime
      inputDim_(dataset->inputDim()),
      hiddenDim_(0), // Hidden dimension might not be applicable here
      outputDim_(dataset->outputDim()),
      sequenceLength_(sequenceLength),
      x(dataset->inputDim(), 1, sequenceLength),
      h(0, 0, sequenceLength), // Hidden buffers may not be needed in this case
      y(dataset->outputDim(), 1, sequenceLength) {
    allocateBuffers();
}

void DataSourceManager::initialize(std::function<void()> onInitialized,
                                   std::function<double(int, double)> inputFunc,
                                   std::function<double(int, double)> targetFunc) {
    // Preserve existing initialization logic or dataset-specific initialization
    if (onInitialized) {
        onInitialized();
    }
}

void DataSourceManager::allocateBuffers() {
    x.allocate_buffers();
    if (hiddenDim_ > 0) {
        h.allocate_buffers();
    }
    y.allocate_buffers();
}

void DataSourceManager::loadSample(int index) {
    if (!dataset_) {
        throw std::runtime_error("❌ Dataset not initialized in DataSourceManager.");
    }

    const auto& inputSample = dataset_->inputAt(index);
    const auto& targetSample = dataset_->targetAt(index);

    if (inputSample.size() != inputDim_ || targetSample.size() != outputDim_) {
        throw std::runtime_error("❌ Data dimensions mismatch in DataSourceManager::loadSample.");
    }

    std::copy(inputSample.begin(), inputSample.end(), x.get_data_buffer_at(0));
    std::copy(targetSample.begin(), targetSample.end(), y.get_data_buffer_at(0));
}

int DataSourceManager::numSamples() const {
    if (!dataset_) {
        throw std::runtime_error("❌ Dataset not initialized in DataSourceManager.");
    }
    return dataset_->numSamples();
}

int DataSourceManager::inputDim() const {
    return inputDim_;
}

int DataSourceManager::outputDim() const {
    return outputDim_;
}

void DataSourceManager::shiftBuffers() {
    x.shift_buffers();
    if (hiddenDim_ > 0) {
        h.shift_buffers();
    }
    y.shift_buffers();
}

void DataSourceManager::randomizeBuffers(double timeOffset) {
    x.randomize_buffers(timeOffset);
    if (hiddenDim_ > 0) {
        h.randomize_buffers(timeOffset);
    }
    y.randomize_buffers(timeOffset);
}
