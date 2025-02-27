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

    // For the recurrent hidden layer.
    DataSource W1;     // Input-to–hidden weights (W_xh)
    DataSource b1;     // Hidden layer biases

    // For the output layer.
    DataSource W2;     // Hidden-to–output weights
    DataSource b2;     // Output layer biases
    
    // Random data sources for initialization.
    DataSource rand1;
    DataSource rand2;
    
    // Asynchronously initialize all DataSources.
    // 'inputFunc' is used to build the input and 'targetFunc' for the target.
    void initialize(std::function<void()> onComplete,
                    double (*inputFunc)(double),
                    double (*targetFunc)(double));
};

#endif // DATASOURCE_MANAGER_H

