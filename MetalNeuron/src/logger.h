#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
public:
    Logger(const std::string& filename);
    ~Logger();
    
    // Logs average errors computed from output and hidden errors across all timesteps.
    void logErrors(const std::vector<float*>& outputErrors, int outputCount, int hiddenCount, int sequenceLength);
    
    // Logs one full iteration's data (inputs, hidden states, outputs, targets) for all timesteps.
    void logRegressionData(const float* output, int outputCount,
                           const float* target, int targetCount);
    
    void logClassificationData(const float* output, int outputCount,
                               const float* target, int targetCount);
    
    void logMSE(float* targetData, float* outputData, int dimension);
    void logCrossEntropyLoss(float* targetData, float* outputData, int dimension);

    void clear();
    
private:
    std::ofstream *logFileStream = nullptr;
    std::string filename_;
};

#endif // LOGGER_H
