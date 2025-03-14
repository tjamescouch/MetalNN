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

// Backward pass for Layer Normalization
kernel void backward_layer_norm(
    device const float* input                [[buffer(0)]],   // [batchSize x featureDim]
    device const float* inputErrors          [[buffer(1)]],   // incoming errors (dY)
    device float* outputErrors               [[buffer(2)]],   // propagated errors (dX)
    device float* gamma                      [[buffer(3)]],   // gamma parameter
    device float* beta                       [[buffer(4)]],   // beta parameter
    device const float* savedMean            [[buffer(5)]],   // mean per sample (forward pass)
    device const float* savedVariance        [[buffer(6)]],   // variance per sample (forward pass)
    constant float& epsilon                  [[buffer(7)]],
    constant int& featureDim                 [[buffer(8)]],
    constant int& batchSize                  [[buffer(9)]],
    constant float& learningRate             [[buffer(10)]],
    uint gid                                 [[thread_position_in_grid]]
) {
    if ((int)gid >= batchSize) return;

    float mean = savedMean[gid];
    float variance = savedVariance[gid];
    float invStd = rsqrt(variance + epsilon);

    // Temporary accumulators
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;

    // Compute intermediate sums for gradients
    for (int f = 0; f < featureDim; f++) {
        int idx = gid * featureDim + f;
        float xhat = (input[idx] - mean) * invStd;
        float dy = inputErrors[idx];
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    // Compute and propagate input gradients, update gamma and beta
    for (int f = 0; f < featureDim; ++f) {
        int idx = gid * featureDim + f;
        float xhat = (input[idx] - mean) * invStd;
        float dy = inputErrors[idx];

        // Input gradient
        outputErrors[idx] = (gamma[f] * invStd / featureDim) *
                            (featureDim * dy - sum_dy - xhat * sum_dy_xhat);

        // Atomically update gamma and beta parameters
        atomic_fetch_add_explicit((device atomic_float*)&gamma[f], -learningRate * dy * xhat / batchSize, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&beta[f], -learningRate * dy / batchSize, memory_order_relaxed);
    }
}
