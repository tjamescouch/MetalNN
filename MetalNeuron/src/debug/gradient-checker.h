//
//  gradient-checker.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//
#ifdef FUTURE_CLASS

#include "neural-engine.h"

class GradientChecker {
public:
    GradientChecker(NeuralEngine* engine, DataSourceManager* dataSource);
    
    float numericalGradient(
        Layer* layer,
        int paramIndex,
        float epsilon = 1e-5f
    );

    void checkLayerGradients(
        Layer* layer,
        float tolerance = 1e-4f
    );

private:
    NeuralEngine* engine_;
    DataSourceManager* dataSource_;

    float computeLoss();
};

#endif
