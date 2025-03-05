#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
public:
    Logger(const std::string& filename, bool isRegression);
    ~Logger();
    
    void clear();
    
    void logErrors(const std::vector<float*>& outputErrors, int outputCount, int hiddenCount, int sequenceLength);
    
    void logAnalytics(const float* output, int outputCount,
                      const float* target, int targetCount);
    
    void logMSE(float* targetData, float* outputData, int dimension);
    void logCrossEntropyLoss(float* targetData, float* outputData, int dimension);


    void logLoss(float loss);
    void accumulateLoss(float loss);
    float finalizeBatchLoss();
    
    void addSample(const float* prediction, const float* target);

    void flushAnalytics();
    
    void clearBatchData();
    void flushBatchData();
    
private:
    void flushRegressionAnalytics();
    void flushClassificationAnalytics();
    
    bool isRegression_ = true;
    float accumulatedLoss_ = 0.0f;
    int numSamples_ = 0;
    std::ofstream *logFileStream = nullptr;
    std::string filename_;
    
    std::vector<std::vector<float>> batchOutputs_;
    std::vector<std::vector<float>> batchTargets_;
    int outputDim_;
};

#endif // LOGGER_H
