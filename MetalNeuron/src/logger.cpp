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

void Logger::logIteration(const float* output, int outputCount,
                          const float* target, int targetCount) {
    std::ofstream logFile(filename_, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    logFile << "clf; hold on;" << std::endl;
    logFile << "ylim([-1.2 1.2]);" << std::endl;

    // Generate x-axis explicitly for this timestep
    logFile << "x = 1:" << outputCount << ";" << std::endl;

    // Log the target and output arrays clearly
    logFile << "target = [ ";
    for (int i = 0; i < targetCount; ++i) {
        logFile << target[i] << (i < targetCount - 1 ? ", " : "");
    }
    logFile << " ];" << std::endl;

    logFile << "output = [ ";
    for (int i = 0; i < outputCount; ++i) {
        logFile << output[i] << (i < outputCount - 1 ? ", " : "");
    }
    logFile << " ];" << std::endl;

    // Plot using scatter plots explicitly
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
