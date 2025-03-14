#pragma once

#include "dataset.h"
#include <functional>
#include <string>

class DataManager {
public:
    DataManager(Dataset* dataset);
    ~DataManager();

    void setDataset(Dataset* dataset);
    Dataset* getCurrentDataset() const;

    void initialize(int batchSize, std::function<void()> callback);

    int inputDim() const;
    int outputDim() const;
    void loadNextBatch(int currentBatchSize);

private:
    Dataset* current_dataset_;
    int sampleIndex_ = 0;

};
