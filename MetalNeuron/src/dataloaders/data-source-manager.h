// DataSourceManager.h
#ifndef DATA_SOURCE_MANAGER_H
#define DATA_SOURCE_MANAGER_H

#include <vector>
#include <functional>

#include "data-source.h"
#include "dataset.h"


class DataSourceManager {
public:
    // Existing constructor remains intact for backward compatibility
    DataSourceManager(int inputDim, int hiddenDim, int outputDim, int sequenceLength);

    // New generalized constructor accepting Dataset interface
    explicit DataSourceManager(Dataset* dataset, int sequenceLength = 1);

    void initialize(std::function<void()> onInitialized,
                    std::function<double(int, double)> inputFunc = nullptr,
                    std::function<double(int, double)> targetFunc = nullptr);

    void loadSample(int index);

    int numSamples() const;
    int inputDim() const;
    int outputDim() const;

    // Existing methods remain unchanged to avoid breaking changes
    void shiftBuffers();
    void randomizeBuffers(double timeOffset);

    // Existing data members (preserved exactly as-is)
    DataSource x;
    DataSource h;
    DataSource y;

private:
    Dataset* dataset_;
    bool ownsDataset_;

    int inputDim_;
    int hiddenDim_;
    int outputDim_;
    int sequenceLength_;

    void allocateBuffers();
};

#endif // DATA_SOURCE_MANAGER_H
