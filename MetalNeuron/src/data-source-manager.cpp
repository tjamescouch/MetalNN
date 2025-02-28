#include "data-source-manager.h"
#include <dispatch/dispatch.h>

DataSourceManager::DataSourceManager(int inputDim, int hiddenDim, int outputDim, int sequenceLength)
    : x(inputDim, 1, sequenceLength),
      y_hat(outputDim, 1, sequenceLength),
      sequenceLength_(sequenceLength)
{
}

DataSourceManager::~DataSourceManager() {}

void DataSourceManager::initialize(std::function<void()> onComplete,
                                   double (*inputFunc)(double, int),
                                   double (*targetFunc)(double, int))
{
    int completedTimesteps = 0;

    for (int t = 0; t < sequenceLength_; ++t) {
        x.buildAsyncAtTimestep(inputFunc, t, [this, t, targetFunc, &completedTimesteps, onComplete]() {
            y_hat.buildAsyncAtTimestep(targetFunc, t, [this, &completedTimesteps, onComplete]() {
                dispatch_async(dispatch_get_main_queue(), ^{
                    completedTimesteps++;
                    if (completedTimesteps == sequenceLength_) {
                        onComplete();
                    }
                });
            });
        });
    }
}

void DataSourceManager::buildInputAtTimestep(std::function<double(double, int)> inputFunc, int timestep, std::function<void()> onComplete) {
    x.buildAsyncAtTimestep(inputFunc, timestep, [onComplete]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            onComplete();
        });
    });
}

void DataSourceManager::buildTargetAtTimestep(std::function<double(double, int)> targetFunc, int timestep, std::function<void()> onComplete) {
    y_hat.buildAsyncAtTimestep(targetFunc, timestep, [onComplete]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            onComplete();
        });
    });
}
