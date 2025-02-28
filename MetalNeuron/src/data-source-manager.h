//
//  data-source-manager.h
//  MetalNeuron
//
//  Created by James Couch on 2025-02-26.
//
#ifndef DATASOURCE_MANAGER_H
#define DATASOURCE_MANAGER_H

#include "data-source.h"
#include <functional>

class DataSourceManager {
public:
    // Constructor: accepts dimensions for input, hidden and output layers.
    DataSourceManager(int inputDim, int hiddenDim, int outputDim);
    ~DataSourceManager();

    // DataSources used in the network.
    DataSource x;      // Input data
    DataSource y_hat;  // Target output for the output layer

    
    // Asynchronously initialize all DataSources.
    // 'inputFunc' is used to build the input and 'targetFunc' for the target.
    void initialize(std::function<void()> onComplete,
                    double (*inputFunc)(double),
                    double (*targetFunc)(double));
};

#endif // DATASOURCE_MANAGER_H

