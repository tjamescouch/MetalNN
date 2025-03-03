//
//  data-manager.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

#pragma once

#include "dataset.h"
#include "data-source-manager.h"

class DataManager {
public:
    DataManager(Dataset* dataset, int sequence_length);

    void initialize(std::function<void()> onInitialized);

    DataSourceManager* getDataSourceManager() const;

    int inputDim() const;
    int outputDim() const;
    int numSamples() const;

    void loadNextSample();

private:
    DataSourceManager* _pDataSourceManager;
    Dataset* _pDataset;
    int sampleIndex_;
};
