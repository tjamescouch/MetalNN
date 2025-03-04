#ifndef DATASET_H
#define DATASET_H

class Dataset {
public:
    virtual ~Dataset() = default;

    // Required methods:
    virtual void loadData() = 0;
    
    virtual float* getInputDataBuffer() = 0;
    virtual float* getTargetDataBuffer() = 0;

    virtual float* getInputDataAt(int timestep) = 0;
    virtual float* getTargetDataAt(int timestep) = 0;
    virtual int numSamples() const = 0;

    virtual int getDatasetSize() const = 0;
    virtual float calculateLoss(const float* predictedData, int outputDim) = 0;


    virtual int inputDim() const = 0;
    virtual int outputDim() const = 0;
    
    virtual void loadSample(int sampleIndex) = 0;
};

#endif
