//
//  data-manger.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

#include "data-manager.h"

DataManager::DataManager(Dataset* dataset, int sequence_length)
: _pDataset(dataset), sampleIndex_(0) {
    _pDataSourceManager = new DataSourceManager(dataset, sequence_length);
}

void DataManager::initialize(std::function<void()> onInitialized) {
    _pDataSourceManager->initialize(onInitialized);
}

DataSourceManager* DataManager::getDataSourceManager() const {
    return _pDataSourceManager;
}

int DataManager::inputDim() const {
    return _pDataset->inputDim();
}

int DataManager::outputDim() const {
    return _pDataset->outputDim();
}

int DataManager::numSamples() const {
    return _pDataset->numSamples();
}

void DataManager::loadNextSample() {
    _pDataSourceManager->loadSample(sampleIndex_);
    sampleIndex_ = (sampleIndex_ + 1) % numSamples();
}
