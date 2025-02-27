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
      y_hat(outputDim, 1),
      W1(inputDim, hiddenDim),
      b1(hiddenDim, 1),
      W2(hiddenDim, outputDim),
      b2(outputDim, 1),
      rand1(inputDim, hiddenDim),
      rand2(hiddenDim, outputDim)
{
}

DataSourceManager::~DataSourceManager() { }

void DataSourceManager::initialize(std::function<void()> onComplete,
                                   double (*inputFunc)(double),
                                   double (*targetFunc)(double))
{
    rand1.initRandomAsync([this, onComplete, inputFunc, targetFunc]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            rand2.initRandomAsync([this, onComplete, inputFunc, targetFunc]() {
                dispatch_async(dispatch_get_main_queue(), ^{
                    x.buildAsync(inputFunc, [this, onComplete, targetFunc]() {
                        dispatch_async(dispatch_get_main_queue(), ^{
                            y_hat.buildAsync(targetFunc, [this, onComplete]() {
                                dispatch_async(dispatch_get_main_queue(), ^{
                                    W1.initRandomAsync([this, onComplete]() {
                                        dispatch_async(dispatch_get_main_queue(), ^{
                                            b1.initRandomAsync([this, onComplete]() {
                                                dispatch_async(dispatch_get_main_queue(), ^{
                                                    W2.initRandomAsync([this, onComplete]() {
                                                        dispatch_async(dispatch_get_main_queue(), ^{
                                                            b2.initRandomAsync([this, onComplete]() {
                                                                dispatch_async(dispatch_get_main_queue(), ^{
                                                                    onComplete();
                                                                });
                                                            });
                                                        });
                                                    });
                                                });
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}
