#pragma once

#include <functional>
#include <string>

#include "dataset.h"
#include "model-config.h"

class DataManager {
public:
    DataManager();
    ~DataManager();

    void setDataset(Dataset* dataset);
    Dataset* getCurrentDataset() const;

    DataManager* configure(ModelConfig* pConfig);
    void initialize(int batchSize, std::function<void()> callback);

    int inputDim() const;
    int outputDim() const;
    void loadNextBatch(int currentBatchSize);

private:
    Dataset* dataset_;
    int sampleIndex_ = 0;

};
