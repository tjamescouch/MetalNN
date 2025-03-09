#include "logger.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include "math-lib.h"

const char* filename = "multilayer_nn_training.m";

Logger::Logger()
:
filename_(filename),
logFileStream(nullptr),
isRegression_(true),
batchSize_(1) {
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

Logger* Logger::instance_ = nullptr;
std::once_flag Logger::initInstanceFlag;

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

void Logger::flushAnalytics() {
    if (isRegression_) {
        return flushRegressionAnalytics();
    }
    
    return flushClassificationAnalytics();
}

void Logger::logAnalytics(const float* output, int outputCount,
                               const float* target, int targetCount) {
    batchOutputs_.emplace_back(output, output + outputCount);
    batchTargets_.emplace_back(target, target + targetCount);
}

void Logger::flushRegressionAnalytics() {
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    *logFileStream << "clf; hold on;" << std::endl;
    

    for (size_t sampleIdx = 0; sampleIdx < batchOutputs_.size(); ++sampleIdx) {
        const auto& output = batchOutputs_[sampleIdx];
        const auto& target = batchTargets_[sampleIdx];
        size_t outputCount = output.size();
        
        *logFileStream << "ylim([-1 1], \"Manual\");" << std::endl;
        *logFileStream << "x = 1:" << outputCount << ";" << std::endl;

        // Log target array
        *logFileStream << "target = [ ";
        for (int i = 0; i < outputCount; ++i)
            *logFileStream << target[i] << (i < outputCount - 1 ? ", " : "");
        *logFileStream << " ];" << std::endl;

        // Log output array
        *logFileStream << "output = [ ";
        for (int i = 0; i < outputCount; ++i)
            *logFileStream << output[i] << (i < outputCount - 1 ? ", " : "");
        *logFileStream << " ];" << std::endl;

        // Plot targets and predictions
        *logFileStream << "scatter(x, target, 'filled', 'b', 'DisplayName', 'Target');" << std::endl;
        *logFileStream << "scatter(x, output, 'filled', 'r', 'DisplayName', 'Prediction');" << std::endl;

        *logFileStream << "legend('show');" << std::endl;
        *logFileStream << "pause(0.01);" << std::endl;
        *logFileStream << "clf; hold on;" << std::endl;
    }

    *logFileStream << "hold off;" << std::endl;
}

void Logger::flushClassificationAnalytics() {
    if (!logFileStream->is_open()) {
        std::cerr << "Error opening log file: " << filename_ << std::endl;
        return;
    }

    
    for (size_t sampleIdx = 0; sampleIdx < batchOutputs_.size(); ++sampleIdx) {
        const auto& output = batchOutputs_[sampleIdx];
        const auto& target = batchTargets_[sampleIdx];
        size_t outputCount = output.size() / batchSize_; //FIXME - just logging the first sample per batch for now

        *logFileStream << "clf; hold on;" << std::endl;
        *logFileStream << "xlabel('Class (Digit)'); ylabel('Probability');" << std::endl;
        *logFileStream << "ylim([0, 1]);" << std::endl;

        *logFileStream << "x = 0:" << (outputCount - 1) << ";" << std::endl;

        // Target vector
        *logFileStream << "target = [";
        for (int i = 0; i < outputCount; ++i) {
            *logFileStream << target[i] << (i < outputCount - 1 ? ", " : "") << " ";
        }
        *logFileStream << "];" << std::endl;

        // Predicted probabilities
        *logFileStream << "output = [";
        for (int i = 0; i < outputCount; ++i) {
            *logFileStream << output[i] << (i < outputCount - 1 ? ", " : "") << " ";
        }
        *logFileStream << "];" << std::endl;

        // Plot as bar plots
        *logFileStream << "bar(x - 0.15, target, 0.3, 'FaceColor', 'b', 'DisplayName', 'Target');" << std::endl;
        *logFileStream << "bar(x + 0.15, output, 0.3, 'FaceColor', 'r', 'DisplayName', 'Prediction');" << std::endl;

        *logFileStream << "legend('show');" << std::endl;
        *logFileStream << "pause(0.05);" << std::endl;
    }

    *logFileStream << "hold off;" << std::endl;
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

void Logger::logLoss(float loss) {
    std::cout << "✨ Loss: " << loss << std::endl;
}

void Logger::accumulateLoss(float loss, int batchSize) {
    accumulatedLoss_ += loss;
    numSamples_+= batchSize;
    logLoss(accumulatedLoss_ / numSamples_);
}

void Logger::finalizeBatchLoss() {
    accumulatedLoss_ = 0.0f;
    numSamples_ = 0;
}

void Logger::clearBatchData() {
    batchOutputs_.clear();
    batchTargets_.clear();
}

void Logger::setBatchSize(int batchSize) {
    batchSize_ = batchSize;
}

Logger& Logger::instance() {
    std::call_once(initInstanceFlag, &Logger::initSingleton);
    return *instance_;
}

void Logger::initSingleton() {
    instance_ = new Logger();
}

void Logger::printFloatBuffer(MTL::Buffer* b, std::string message) {
    float* data = static_cast<float*>(b->contents());
    size_t numFloats = b->length() / sizeof(float);
    
    std::cout << message << " => [";
    for (int i = 0; i < numFloats; ++i) {
        std::cout << data[i];
        if (i < numFloats - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
