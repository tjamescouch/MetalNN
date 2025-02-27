//
//  logger.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger {
public:
    // Constructs a logger that writes to the given file.
    Logger(const std::string& filename);
    ~Logger();

    // Logs the average errors computed from the output and hidden error buffers.
    // 'outputError' contains 'outputCount' floats, and 'hiddenError' contains 'hiddenCount' floats.
    void logErrors(const float* outputError, int outputCount, const float* hiddenError, int hiddenCount);

    // Logs one iteration's data (input, hidden, output, and target arrays) to the log file.
    void logIteration(const float* input,  int inputCount,
                      const float* hidden, int hiddenCount,
                      const float* output, int outputCount,
                      const float* target, int targetCount);

    // Clears the log file.
    void clear();

private:
    std::string filename_;
};

#endif // LOGGER_H
