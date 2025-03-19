//
//  data-manger.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include <stdexcept>

#include "data-manager.h"
#include <cassert>
#include "math-lib.h"
#include "function-dataset.h"
#include "mnist-dataset.h"


DataManager::DataManager()
: dataset_(nullptr) {
}

DataManager* DataManager::configure(ModelConfig* pConfig) {
    if (pConfig->dataset.type == "mnist") {
        dataset_ = new MNISTDataset(
                                    pConfig->dataset.images,
                                    pConfig->dataset.labels
                                    );
    } else if (pConfig->dataset.type == "function") {
        int inputShape[2] = {};
        pConfig->layers.front().params.at("output_shape").get_value_inplace(inputShape);
        int sequenceLength = inputShape[0];
        int featureDim = inputShape[1];

        int outputDim = 0;
        if (pConfig->layers.back().params.contains("output_shape")) {
            int outputShape[2] = {};
            pConfig->layers.back().params.at("output_shape").get_value_inplace(outputShape);
            outputDim = outputShape[1];
        } else {
            outputDim = pConfig->layers.back().params.at("output_size").get_value<int>();
        }

        dataset_ = new FunctionDataset(mathlib::inputFunc, mathlib::targetFunc,
                                       sequenceLength,
                                       featureDim,
                                       outputDim,
                                       1000); // Arbitrary dataset size
    } else {
        throw std::runtime_error("Unsupported dataset type: " + pConfig->dataset.type);
    }
    
    return this;
}

DataManager::~DataManager() {
    if (dataset_) {
        delete dataset_;
        dataset_ = nullptr;
    }
}

void DataManager::setDataset(Dataset* dataset) {
    if (dataset_) {
        delete dataset_;
    }
    dataset_ = dataset;
}

Dataset* DataManager::getCurrentDataset() const {
    if (!dataset_) {
        throw std::runtime_error("Dataset has not been set.");
    }
    return dataset_;
}

void DataManager::initialize(int batchSize, std::function<void()> callback) {
    if (!dataset_) {
        throw std::runtime_error("Cannot initialize DataManager: no dataset set.");
    }
    
    dataset_->loadData(batchSize);
    callback();
}

int DataManager::inputDim() const {
    return dataset_->inputDim();
}

int DataManager::outputDim() const {
    return dataset_->outputDim();
}

void DataManager::loadNextBatch(int currentBatchSize) {
    dataset_->loadNextBatch(currentBatchSize);
}
