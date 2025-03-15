#include <metal_stdlib>

#include "common.metal"

using namespace metal;


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);



#include <metal_stdlib>
using namespace metal;

// Forward pass kernel for Layer Normalization
kernel void forward_layer_norm(
    device const float* input            [[buffer(0)]],    // [batchSize x featureDim]
    device float* output                 [[buffer(1)]],
    device const float* gamma            [[buffer(2)]],
    device const float* beta             [[buffer(3)]],
    device float* savedMean              [[buffer(4)]],
    device float* savedVariance          [[buffer(5)]],
    constant float& epsilon              [[buffer(6)]],
    constant int& featureDim             [[buffer(7)]],
    constant int& batchSize              [[buffer(8)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if ((int)gid >= batchSize) return;

    // Compute mean per sample
    float mean = 0.0f;
    for (int f = 0; f < featureDim; f++) {
        mean += input[gid * featureDim + f];
    }
    mean /= float(featureDim);
    savedMean[gid] = mean;

    // Compute variance per sample (fixed here explicitly)
    float variance = 0.0f;
    for (int f = 0; f < featureDim; f++) {
        float val = input[gid * featureDim + f] - mean;
        variance += val * val;
    }
    variance /= float(featureDim);
    savedVariance[gid] = variance;

    float invStd = rsqrt(variance + epsilon);

    // Normalize explicitly each feature for this sample
    for (int f = 0; f < featureDim; f++) {
        int idx = gid * featureDim + f;
        float norm = (input[idx] - mean) * invStd;
        output[idx] = norm * gamma[f] + beta[f];
    }
}

kernel void backward_layer_norm(
    device const float* input                [[buffer(0)]],   // [batchSize x featureDim]
    device const float* inputErrors          [[buffer(1)]],   // incoming errors (dY)
    device float* outputErrors               [[buffer(2)]],   // propagated errors (dX)
    device float* gamma                      [[buffer(3)]],   // gamma parameter
    device float* beta                       [[buffer(4)]],   // beta parameter
    device const float* savedMean            [[buffer(5)]],   // mean per sample (forward pass)
    device const float* savedVariance        [[buffer(6)]],   // variance per sample (forward pass)
    constant float& epsilon                  [[buffer(7)]],
    constant uint& featureDim                [[buffer(8)]],
    constant uint& batchSize                 [[buffer(9)]],
    constant float& learningRate             [[buffer(10)]],
    device atomic_float* gradientsBeta       [[buffer(14)]],
    device atomic_float* gradientsGamma      [[buffer(15)]],
    uint gid                               [[thread_position_in_grid]]
) {
    if (gid >= batchSize) return;

    float mean = savedMean[gid];
    float variance = savedVariance[gid];
    float invStd = rsqrt(variance + 1e-5f);

    // Accumulate gradients for gamma and beta
    for (uint featureIdx = 0; featureIdx < featureDim; ++featureIdx) {
        uint idx = gid * featureDim + featureIdx;

        float xhat = (input[idx] - mean) * invStd;
        float dy = inputErrors[idx];

        // Accumulate gamma and beta gradients
        atomic_fetch_add_explicit(&gradientsGamma[featureIdx], dy * xhat, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradientsBeta[featureIdx], dy, memory_order_relaxed);
    }

    // Compute gradients for input activations
    float sumDyGammaXhat = 0.0f;
    float sumDyGamma = 0.0f;

    for (uint featureIdx = 0; featureIdx < featureDim; ++featureIdx) {
        uint idx = gid * featureDim + featureIdx;
        float xhat = (input[idx] - savedMean[gid]) * invStd;
        float dy = inputErrors[idx];

        sumDyGammaXhat += dy * gamma[featureIdx] * xhat;
        sumDyGamma += dy * gamma[featureIdx];
    }

    for (uint featureIdx = 0; featureIdx < featureDim; ++featureIdx) {
        uint idx = gid * featureDim + featureIdx;

        float dy = outputErrors[idx];
        float xhat = (input[idx] - savedMean[gid]) * invStd;

        outputErrors[idx] = (gamma[featureIdx] * invStd / featureDim) *
                           (featureDim * dy - sumDyGamma - xhat * sumDyGammaXhat);
    }
}

