//
//  mnist-dataset.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

#include "mnist-dataset.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <mach-o/dyld.h>

MNISTDataset::MNISTDataset(const std::string& imagesFilename, const std::string& labelsFilename) {
    namespace fs = std::filesystem;

    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        throw std::runtime_error("❌ Executable path buffer too small.");
    }

    fs::path executablePath = fs::canonical(path);
    fs::path resourcesPath = executablePath.parent_path().parent_path() / "Resources";

    fs::path imagesPath = resourcesPath / imagesFilename;
    fs::path labelsPath = resourcesPath / labelsFilename;

    if (!fs::exists(imagesPath)) {
        throw std::runtime_error("❌ MNIST images file not found at: " + imagesPath.string());
    }
    if (!fs::exists(labelsPath)) {
        throw std::runtime_error("❌ MNIST labels file not found at: " + labelsPath.string());
    }

    loadImages(imagesPath.string());
    loadLabels(labelsPath.string());
}

void MNISTDataset::loadData() {
    // Data is already loaded in constructor, nothing else required here.
}

int MNISTDataset::numSamples() const {
    return static_cast<int>(inputs_.size());
}

int MNISTDataset::inputDim() const {
    return 784; // 28x28 images flattened
}

int MNISTDataset::outputDim() const {
    return 10; // Digits 0-9
}

const std::vector<float>& MNISTDataset::inputAt(int index) {
    return inputs_[index];
}

const std::vector<float>& MNISTDataset::targetAt(int index) {
    return targets_[index];
}

float* MNISTDataset::getInputDataAt(int timestep) {
    return inputs_[timestep].data();
}

float* MNISTDataset::getTargetDataAt(int timestep) {
    return targets_[timestep].data();
}

int MNISTDataset::getDatasetSize() const {
    return static_cast<int>(inputs_.size());
}

static int32_t readBigEndianInt(std::ifstream& file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    return __builtin_bswap32(value);
}

void MNISTDataset::loadImages(const std::string& imagesPath) {
    std::ifstream file(imagesPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("❌ Cannot open images file at: " + imagesPath);
    }

    int32_t magic = readBigEndianInt(file);
    int32_t numImages = readBigEndianInt(file);
    int32_t rows = readBigEndianInt(file);
    int32_t cols = readBigEndianInt(file);

    if (magic != 2051) {
        throw std::runtime_error("❌ Invalid MNIST image file magic number.");
    }

    inputs_.resize(numImages, std::vector<float>(rows * cols));

    for (int i = 0; i < numImages; ++i) {
        std::vector<unsigned char> imageData(rows * cols);
        file.read(reinterpret_cast<char*>(imageData.data()), rows * cols);
        for (int px = 0; px < rows * cols; ++px) {
            inputs_[i][px] = imageData[px] / 255.0f; // Normalize pixel values
        }
    }
}

void MNISTDataset::loadLabels(const std::string& labelsPath) {
    std::ifstream file(labelsPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("❌ Cannot open labels file at: " + labelsPath);
    }

    int32_t magic = readBigEndianInt(file);
    int32_t numLabels = readBigEndianInt(file);

    if (magic != 2049) {
        throw std::runtime_error("❌ Invalid MNIST label file magic number.");
    }

    targets_.resize(numLabels, std::vector<float>(10, 0.0f));

    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        targets_[i][label] = 1.0f;  // One-hot encode labels
    }
}

float MNISTDataset::calculateLoss(const float* predictedData, int outputDim) {
    const float epsilon = 1e-10f;
    const float* targetData = getTargetDataAt(0);  // Assuming current timestep
    float loss = 0.0f;

    for (int i = 0; i < outputDim; ++i) {
        if (targetData[i] > 0.5f) { // one-hot target
            loss = -logf(predictedData[i] + epsilon);
            break;
        }
    }

    return loss;
}
/*
void MNISTDataset::loadSample(int sampleIndex) {
    float* inputBuffer = getInputDataBuffer();
    float* targetBuffer = getTargetDataBuffer();

    const std::vector<float>& image = images_[sampleIndex];
    const std::vector<float>& target = targets_[sampleIndex];

    std::copy(image.begin(), image.end(), inputBuffer);
    std::copy(target.begin(), target.end(), targetBuffer);
}
*/
void MNISTDataset::loadSample(int sampleIndex) {
    float* inputBuffer = inputs_[sampleIndex].data();
    float* targetBuffer = targets_[sampleIndex].data();

    std::memcpy(getInputDataBuffer(), inputBuffer, sizeof(float) * inputDim());
    std::memcpy(getTargetDataBuffer(), targetBuffer, sizeof(float) * outputDim());
}

// Returns pointer to the buffer for current input data
float* MNISTDataset::getInputDataBuffer() {
    return inputs_[0].data();
}

// Returns pointer to the buffer for current target data
float* MNISTDataset::getTargetDataBuffer() {
    return targets_[0].data();
}


