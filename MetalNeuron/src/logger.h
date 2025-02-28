#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>

class Logger {
public:
    Logger(const std::string& filename);
    ~Logger();
    
    // Logs average errors computed from output and hidden errors across all timesteps.
    void logErrors(const std::vector<float*>& outputErrors, int outputCount,
                   const std::vector<float*>& hiddenErrors, int hiddenCount, int sequenceLength);
    
    // Logs one full iteration's data (inputs, hidden states, outputs, targets) for all timesteps.
    void logIteration(const std::vector<float*>& outputs, int outputCount,
                      const std::vector<float*>& targets, int targetCount,
                      int sequenceLength);
    
    void clear();
    
private:
    std::string filename_;
};

#endif // LOGGER_H
