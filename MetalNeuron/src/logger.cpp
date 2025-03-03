#include "logger.h"
#include "common.h"
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

void Logger::logErrors(const std::vector<float*>& outputErrors, int outputCount, int hiddenCount, int sequenceLength) {
    float avgOutputError = 0.0f;
    
    for (int t = 0; t < sequenceLength; ++t) {
        float timestepOutputError = 0.0f;
        
        for (int i = 0; i < outputCount; ++i)
            timestepOutputError += fabs(outputErrors[t][i]);
        timestepOutputError /= outputCount;
        
        avgOutputError += timestepOutputError;
    }
    
    avgOutputError /= sequenceLength;
    
    std::cout << "AVG OUTPUT ERROR (across sequence): " << avgOutputError << std::endl;
}

void Logger::logRegressionData(const float* output, int outputCount,
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

void Logger::logClassificationData(const float* output, int outputCount,
                                   const float* target, int targetCount) {
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    // Clear figure and prepare plot
    *logFileStream << "clf; hold on;" << std::endl;
    *logFileStream << "xlabel('Class (Digit)'); ylabel('Probability');" << std::endl;
    *logFileStream << "ylim([0, 1]);" << std::endl;
    
    // X-axis: classes 0-9
    *logFileStream << "x = 0:" << (outputCount - 1) << ";" << std::endl;
    
    // Target vector
    *logFileStream << "target = [";
    for (int i = 0; i < targetCount; ++i) {
        *logFileStream << target[i] << (i < targetCount - 1 ? ", " : "") << std::endl;
    }
    *logFileStream << "];";
    
    // Predicted probabilities
    *logFileStream << "output = [";
    for (int i = 0; i < outputCount; ++i) {
        *logFileStream << output[i] << (i < outputCount - 1 ? ", " : "") << std::endl;
    }
    *logFileStream << "];";
    
    // Bar plot for clear visual comparison
    *logFileStream << "bar(x - 0.15, target, 0.3, 'FaceColor', 'b', 'DisplayName', 'Target');" << std::endl;
    *logFileStream << "bar(x + 0.15, output, 0.3, 'FaceColor', 'r', 'DisplayName', 'Prediction');" << std::endl;
    *logFileStream << "legend('show');" << std::endl;
    *logFileStream << "hold off; pause(0.05);" << std::endl;
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

void Logger::logMSE(float* targetData, float* outputData, int dimension) {
    float mse = 0.0f;
    for (int i = 0; i < dimension; ++i) {
        float diff = targetData[i] - outputData[i];
        mse += diff * diff;
    }
    mse /= dimension;
    std::printf("Mean Squared Error: %f\n", mse);
}

void Logger::logCrossEntropyLoss(float *targetData, float *outputData, int dimension) {
    float epsilon = 1e-10f; // Numerical stability
    float loss = 0.0f;

    // Cross-entropy loss for one-hot targets
    for (int i = 0; i < dimension; ++i) {
        if (targetData[i] > 0.5f) {  // Exactly one entry is 1.0
            loss = -logf(std::max(outputData[i], epsilon));
            break;
        }
    }
    std::printf("Cross Entropy Loss: %f\n", loss);

#ifdef DEBUG_CROSS_ENTROPY_LOSS
    float softmaxSum = 0.0f;
    for (int i = 0; i < dimension; ++i) {
        softmaxSum += outputData[i];
    }
    std::printf("Softmax sum: %f\n", softmaxSum);

    std::printf("Targets vs Predictions:\n");
    for (int i = 0; i < dimension; ++i) {
        std::printf("Target[%d]: %.2f, Predicted[%d]: %.4f\n", i, targetData[i], i, outputData[i]);
    }
#endif
}
