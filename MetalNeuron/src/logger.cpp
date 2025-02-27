//
//  logger.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-27.
//

#include "logger.h"
#include <fstream>
#include <iostream>
#include <cmath>

Logger::Logger(const std::string& filename)
    : filename_(filename) {
}

Logger::~Logger() {
}

void Logger::logErrors(const float* outputError, int outputCount, const float* hiddenError, int hiddenCount) {
    float avgOutput = 0.0f;
    for (int i = 0; i < outputCount; i++) {
        avgOutput += outputError[i];
    }
    avgOutput /= outputCount;

    float avgHidden = 0.0f;
    for (int i = 0; i < hiddenCount; i++) {
        avgHidden += hiddenError[i];
    }
    avgHidden /= hiddenCount;

    std::cout << "AVG OUTPUT ERROR: " << fabs(avgOutput) << std::endl;
    std::cout << "AVG HIDDEN ERROR: " << fabs(avgHidden) << std::endl;
}

void Logger::logIteration(const float* input, int inputCount,
                            const float* hidden, int hiddenCount,
                            const float* output, int outputCount,
                            const float* target, int targetCount) {
    std::ofstream logFile(filename_, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }
    logFile << "clf; hold on;" << std::endl;
    logFile << "ylim([-1 1]);" << std::endl;

    logFile << "# Logging iteration" << std::endl;

    logFile << "x = [ ";
    for (int i = 0; i < inputCount; i++) {
        if (i > 0)
            logFile << ", ";
        logFile << i;
    }
    logFile << " ]" << std::endl;

    logFile << "input = [ ";
    for (int i = 0; i < inputCount; i++) {
        if (i > 0)
            logFile << ", ";
        logFile << input[i];
    }
    logFile << " ]" << std::endl;

    logFile << "hidden = [ ";
    for (int i = 0; i < hiddenCount; i++) {
        if (i > 0)
            logFile << ", ";
        logFile << hidden[i];
    }
    logFile << " ]" << std::endl;

    logFile << "output = [ ";
    for (int i = 0; i < outputCount; i++) {
        if (i > 0)
            logFile << ", ";
        logFile << output[i];
    }
    logFile << " ]" << std::endl;

    logFile << "target = [ ";
    for (int i = 0; i < targetCount; i++) {
        if (i > 0)
            logFile << ", ";
        logFile << target[i];
    }
    logFile << " ]" << std::endl;

    logFile << "scatter(1:length(input), input, 'b');" << std::endl;
    logFile << "scatter(1:length(output), output, 'r');" << std::endl;
    logFile << "hold off; pause(0.01);" << std::endl;
    logFile.close();
}

void Logger::clear() {
    std::ofstream logFile(filename_, std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }
    logFile.close();
}
