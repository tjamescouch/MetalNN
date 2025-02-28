#include "logger.h"
#include <fstream>
#include <iostream>
#include <cmath>

Logger::Logger(const std::string& filename)
    : filename_(filename) {
}

Logger::~Logger() {
}

void Logger::logErrors(const std::vector<float*>& outputErrors, int outputCount,
                       const std::vector<float*>& hiddenErrors, int hiddenCount, int sequenceLength) {
    float avgOutputError = 0.0f;
    float avgHiddenError = 0.0f;

    for (int t = 0; t < sequenceLength; ++t) {
        float timestepOutputError = 0.0f;
        float timestepHiddenError = 0.0f;

        for (int i = 0; i < outputCount; ++i)
            timestepOutputError += fabs(outputErrors[t][i]);
        timestepOutputError /= outputCount;

        for (int i = 0; i < hiddenCount; ++i)
            timestepHiddenError += fabs(hiddenErrors[t][i]);
        timestepHiddenError /= hiddenCount;

        avgOutputError += timestepOutputError;
        avgHiddenError += timestepHiddenError;
    }

    avgOutputError /= sequenceLength;
    avgHiddenError /= sequenceLength;

    std::cout << "AVG OUTPUT ERROR (across sequence): " << avgOutputError << std::endl;
    std::cout << "AVG HIDDEN ERROR (across sequence): " << avgHiddenError << std::endl;
}

void Logger::logIteration(const std::vector<float*>& outputs, int outputCount,
                          const std::vector<float*>& targets, int targetCount,
                          int sequenceLength) {
    std::ofstream logFile(filename_, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    logFile << "clf; hold on;" << std::endl;
    logFile << "ylim([-1.2 1.2]);" << std::endl;

    // Concatenate outputs and targets across timesteps for clearer visualization
    logFile << "output = [ ";
    for (int t = 0; t < sequenceLength; ++t) {
        for (int i = 0; i < outputCount; ++i) {
            logFile << outputs[t][i];
            if (!(t == sequenceLength - 1 && i == outputCount - 1))
                logFile << ", ";
        }
    }
    logFile << " ];" << std::endl;

    logFile << "target = [ ";
    for (int t = 0; t < sequenceLength; ++t) {
        for (int i = 0; i < targetCount; ++i) {
            logFile << targets[t][i];
            if (!(t == sequenceLength - 1 && i == targetCount - 1))
                logFile << ", ";
        }
    }
    logFile << " ];" << std::endl;

    // Generate the x-axis values (single continuous range)
    int totalPoints = outputCount * sequenceLength;
    logFile << "x = 1:" << totalPoints << ";" << std::endl;

    // Plot clearly labeled predictions and targets
    logFile << "scatter(x, target, 'filled', 'b', 'DisplayName', 'Target');" << std::endl;
    logFile << "scatter(x, output, 'filled', 'r', 'DisplayName', 'Prediction');" << std::endl;

    logFile << "legend('show');" << std::endl;

    logFile << "hold off; pause(0.01);" << std::endl;

    logFile.close();
}


void Logger::clear() {
    std::ofstream logFile(filename_, std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "Error clearing log file: " << filename_ << std::endl;
        return;
    }
    logFile.close();
}
