#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
public:
    Logger(const std::string& filename, bool isRegression);
    ~Logger();
    
    void logErrors(const std::vector<float*>& outputErrors, int outputCount, int hiddenCount, int sequenceLength);
    
    void logAnalytics(const float* output, int outputCount,
                      const float* target, int targetCount);
    
    void logRegressionData(const float* output, int outputCount,
                           const float* target, int targetCount);
    
    void logClassificationData(const float* output, int outputCount,
                               const float* target, int targetCount);
    
    void logMSE(float* targetData, float* outputData, int dimension);
    void logCrossEntropyLoss(float* targetData, float* outputData, int dimension);

    void clear();
    void logLoss(float loss);
    
    bool isRegression_ = true;
    
private:
    std::ofstream *logFileStream = nullptr;
    std::string filename_;
};

#endif // LOGGER_H
