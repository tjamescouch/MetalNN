#include "logger.h"
#include <fstream>
#include <iostream>
#include <cmath>

Logger::Logger(const std::string& filename)
: filename_(filename), logFileStream(nullptr) {
    logFileStream = new std::ofstream(filename_, std::ios::app);
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
    }
}

Logger::~Logger() {
    if (logFileStream) {
        if (logFileStream->is_open())
            logFileStream->close();
        delete logFileStream;
        logFileStream = nullptr;
    }
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
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }
    
    *logFileStream << "clf; hold on;" << std::endl;
    *logFileStream << "ylim([-1.2 1.2]);" << std::endl;
    
    // Generate x-axis explicitly for this timestep
    *logFileStream << "x = 1:" << outputCount << ";" << std::endl;
    
    // Log the target and output arrays clearly
    *logFileStream << "target = [ ";
    for (int i = 0; i < targetCount; ++i) {
        *logFileStream << target[i] << (i < targetCount - 1 ? ", " : "");
    }
    *logFileStream << " ];" << std::endl;
    
    *logFileStream << "output = [ ";
    for (int i = 0; i < outputCount; ++i) {
        *logFileStream << output[i] << (i < outputCount - 1 ? ", " : "");
    }
    *logFileStream << " ];" << std::endl;
    
    // Plot using scatter plots explicitly
    *logFileStream << "scatter(x, target, 'filled', 'b', 'DisplayName', 'Target');" << std::endl;
    *logFileStream << "scatter(x, output, 'filled', 'r', 'DisplayName', 'Prediction');" << std::endl;
    
    *logFileStream << "legend('show');" << std::endl;
    *logFileStream << "hold off; pause(0.01);" << std::endl;
    

}



void Logger::clear() {
    // Close the existing member stream if open.
    if (logFileStream && logFileStream->is_open()) {
        logFileStream->close();
    }
    
    {
        std::ofstream ofs(filename_, std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "Error clearing log file: " << filename_ << std::endl;
            return;
        }
    }
    
    logFileStream->open(filename_, std::ios::app);
    if (!logFileStream->is_open()) {
        std::cerr << "Error reopening log file: " << filename_ << std::endl;
    }
}
