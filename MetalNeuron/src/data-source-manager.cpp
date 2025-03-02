#include "data-source-manager.h"
#include <dispatch/dispatch.h>

DataSourceManager::DataSourceManager(int inputDim, int hiddenDim, int outputDim, int sequenceLength)
    : x(inputDim, 1, sequenceLength),
      y(outputDim, 1, sequenceLength),
      sequenceLength_(sequenceLength)
{
}

DataSourceManager::~DataSourceManager() {}

void DataSourceManager::initialize(std::function<void()> onComplete,
                                   double (*inputFunc)(double, double),
                                   double (*targetFunc)(double, double))
{
    for (int t = 0; t < sequenceLength_; ++t) {
        x.buildAsyncAtTimestep(inputFunc, t, [this, t, targetFunc, onComplete]() {
            y.buildAsyncAtTimestep(targetFunc, t, [this, onComplete]() {
                dispatch_async(dispatch_get_main_queue(), ^{
                    completedTimesteps++;
                    if (completedTimesteps.load() == sequenceLength_) {
                        onComplete();
                    }
                });
            });
        });
    }
}

void DataSourceManager::buildInputAtTimestep(std::function<double(double, double)> inputFunc, int timestep, std::function<void()> onComplete) {
    x.buildAsyncAtTimestep(inputFunc, timestep, [onComplete]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            onComplete();
        });
    });
}

void DataSourceManager::buildTargetAtTimestep(std::function<double(double, double)> targetFunc, int timestep, std::function<void()> onComplete) {
    y.buildAsyncAtTimestep(targetFunc, timestep, [onComplete]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            onComplete();
        });
    });
}
