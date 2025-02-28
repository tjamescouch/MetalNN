//
//  data-source-manager.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-02-26.
//

#include "data-source-manager.h"
#include <dispatch/dispatch.h>

DataSourceManager::DataSourceManager(int inputDim, int hiddenDim, int outputDim)
: x(inputDim, 1),
y_hat(outputDim, 1)
{
}

DataSourceManager::~DataSourceManager() { }

void DataSourceManager::initialize(std::function<void()> onComplete,
                                   double (*inputFunc)(double),
                                   double (*targetFunc)(double))
{
    x.buildAsync(inputFunc, [this, onComplete, targetFunc]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            y_hat.buildAsync(targetFunc, [this, onComplete]() {
                onComplete();
            });
        });
    });
}
