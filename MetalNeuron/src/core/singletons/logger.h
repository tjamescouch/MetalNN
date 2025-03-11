#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>
#include <mutex>

namespace MTL {
class Buffer;
}

class Logger {
public:
    static Logger& instance();
    
    void clear();
    
    void logErrors(const std::vector<float*>& outputErrors, int outputCount, int hiddenCount, int sequenceLength);
    
    void logAnalytics(const float* output, int outputCount,
                      const float* target, int targetCount);
    
    void logMSE(float* targetData, float* outputData, int dimension);
    void logCrossEntropyLoss(float* targetData, float* outputData, int dimension);


    void logLoss(float loss);
    void accumulateLoss(float loss, int currentBatchSize);
    void finalizeBatchLoss();
    
    void addSample(const float* prediction, const float* target);

    void flushAnalytics();
    
    void clearBatchData();
    void flushBatchData();
    void setBatchSize(int batchSize);
    void setIsRegression(bool isRegression) { isRegression_ = isRegression; }
    
    void printFloatBuffer(MTL::Buffer* b, std::string message);
    void printFloatBuffer(MTL::Buffer* b, std::string message, int maxElements);
    
    void printFloatBufferL2Norm(MTL::Buffer* b, std::string message);
    void printFloatBufferMeanL2Norm(MTL::Buffer* b, std::string message);
    
private:
    void flushRegressionAnalytics();
    void flushClassificationAnalytics();
    
    bool isRegression_ = true;
    float accumulatedLoss_ = 0.0f;
    int numSamples_ = 0;
    int batchSize_ = 1;
    
    std::ofstream *logFileStream = nullptr;
    std::string filename_;
    
    std::vector<std::vector<float>> batchOutputs_;
    std::vector<std::vector<float>> batchTargets_;
    int outputDim_;

    Logger();
    ~Logger();
    static void initSingleton();
    
    static Logger* instance_;
    static std::once_flag initInstanceFlag;
};

#endif // LOGGER_H
