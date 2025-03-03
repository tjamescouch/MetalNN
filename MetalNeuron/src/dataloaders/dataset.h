//
//  dataset.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

// Dataset.h
#ifndef DATASET_H
#define DATASET_H

#include <vector>

class Dataset {
public:
    virtual ~Dataset() {}

    virtual int numSamples() const = 0;
    virtual int inputDim() const = 0;
    virtual int outputDim() const = 0;

    virtual const std::vector<float>& inputAt(int index) = 0;
    virtual const std::vector<float>& targetAt(int index) = 0;
};

#endif // DATASET_H
