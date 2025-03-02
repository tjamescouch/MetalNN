//
//  gradient-checker.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-02.
//

#ifndef GRADIENT_CHECKER_H
#define GRADIENT_CHECKER_H

#include "../neural-engine.h"
#include "../dataloaders/data-source-manager.h"

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

#endif // GRADIENT_CHECKER_H
