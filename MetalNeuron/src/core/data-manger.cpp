//
//  data-manger.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#include <stdexcept>

#include "data-manager.h"
#include <cassert>


DataManager::DataManager(Dataset* dataset)
: current_dataset_(dataset) {
}

DataManager::~DataManager() {
    if (current_dataset_) delete current_dataset_;
}

void DataManager::setDataset(Dataset* dataset) {
    if (current_dataset_) delete current_dataset_;
    current_dataset_ = dataset;
}

Dataset* DataManager::getCurrentDataset() const {
    if (!current_dataset_) {
        throw std::runtime_error("Dataset has not been set.");
    }
    return current_dataset_;
}

void DataManager::initialize(std::function<void()> callback) {
    if (!current_dataset_) {
        throw std::runtime_error("Cannot initialize DataManager: no dataset set.");
    }

    current_dataset_->loadData();
    callback();
}

int DataManager::inputDim() const {
    return current_dataset_->inputDim();
}

int DataManager::outputDim() const {
    return current_dataset_->outputDim();
}

void DataManager::loadNextSample() {
    current_dataset_->loadNextSample();  // Delegate responsibility directly to the dataset
}

void DataManager::loadNextBatch(int batchSize) {
    for (int i = 0; i < batchSize; ++i) {
        current_dataset_->loadNextSample();  // Repeatedly load samples via the dataset's own indexing mechanism
    }
}
