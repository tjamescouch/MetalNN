//
//  mnist-data-loader.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

#include "mnist-data-loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <mach-o/dyld.h>

MNISTDataLoader::MNISTDataLoader(const std::string& imagesFilename, const std::string& labelsFilename) {
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

int MNISTDataLoader::numSamples() const {
    return (int)inputs_.size();
}

int MNISTDataLoader::inputDim() const {
    return 784; // 28x28 images flattened
}

int MNISTDataLoader::outputDim() const {
    return 10; // Digits 0-9
}

const std::vector<float>& MNISTDataLoader::inputAt(int index) {
    return inputs_[index];
}

const std::vector<float>& MNISTDataLoader::targetAt(int index) {
    return targets_[index];
}

static int32_t readBigEndianInt(std::ifstream& file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    return __builtin_bswap32(value);
}

void MNISTDataLoader::loadImages(const std::string& imagesPath) {
    std::ifstream file(imagesPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("❌ Cannot open images file at: " + imagesPath);
    }

    int32_t magic = readBigEndianInt(file);
    int32_t numImages = readBigEndianInt(file);
    int32_t rows = readBigEndianInt(file);
    int32_t cols = readBigEndianInt(file);

    std::cout << "Image file header: magic = " << magic
              << ", numImages = " << numImages
              << ", rows = " << rows
              << ", cols = " << cols << std::endl;

    if (magic != 2051) {
        throw std::runtime_error("❌ Invalid MNIST image file magic number.");
    }

    inputs_.resize(numImages, std::vector<float>(rows * cols));

    for (int i = 0; i < numImages; ++i) {
        std::vector<unsigned char> imageData(rows * cols);
        file.read(reinterpret_cast<char*>(imageData.data()), rows * cols);
        for (int px = 0; px < rows * cols; ++px) {
            inputs_[i][px] = imageData[px] / 255.0f;
        }
#ifdef DEBUG_MNIST_LOADER
        std::cout << "First MNIST image loaded pixels: ";
        for (int px = 0; px < rows * cols; ++px) {
            std::cout << inputs_[i][px] << " ";
        }
        std::cout << std::endl;
#endif
    }
}

void MNISTDataLoader::loadLabels(const std::string& labelsPath) {
    std::ifstream file(labelsPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open labels file.");
    }

    int32_t magic, numLabels;
    file.read((char*)&magic, 4);
    file.read((char*)&numLabels, 4);
    magic = __builtin_bswap32(magic);
    numLabels = __builtin_bswap32(numLabels);

    if (magic != 2049) {
        throw std::runtime_error("❌ Invalid MNIST label file magic number.");
    }

    targets_.resize(numLabels, std::vector<float>(10, 0.0f));

    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read((char*)&label, 1);
        targets_[i][label] = 1.0f;  // Exactly one entry set to 1.0, rest 0.0

        if (i == 0) {
            std::cout << "First MNIST label loaded (one-hot): ";
            for (int k = 0; k < 10; ++k) {
                std::cout << targets_[0][k] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Labels magic number: " << magic << ", num_labels: " << numLabels << std::endl;
}
