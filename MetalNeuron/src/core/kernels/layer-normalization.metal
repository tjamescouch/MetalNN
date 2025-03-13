#include <metal_stdlib>

#include "common.metal"

using namespace metal;

#define REDUCTION_SUM     0
#define REDUCTION_MEAN    1
#define REDUCTION_MAX     2
#define REDUCTION_MIN     3
#define REDUCTION_SOFTMAX 4


#define ACTIVATION_LINEAR  0
#define ACTIVATION_RELU    1
#define ACTIVATION_TANH    2
#define ACTIVATION_SIGMOID 3
#define ACTIVATION_SOFTMAX 4


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
    device const float*  input           [[buffer(0)]],   // [batchSize x featureDim]
    device const float*  inputErrors     [[buffer(1)]],   // dY coming into LayerNorm (same shape)
    device float*        outputErrors    [[buffer(2)]],   // dX going out of LayerNorm
    device float*        gamma           [[buffer(3)]],   // Learnable parameter gamma
    device float*        beta            [[buffer(4)]],   // Learnable parameter beta
    device const float*  savedMean       [[buffer(5)]],   // Per-sample saved mean
    device const float*  savedVariance   [[buffer(6)]],   // Per-sample saved variance
    constant float&      epsilon         [[buffer(7)]],
    constant int&        featureDim      [[buffer(8)]],
    constant int&        batchSize       [[buffer(9)]],
    constant float&      learningRate    [[buffer(10)]],
    uint                 gid             [[thread_position_in_grid]]
) {
    if ((int)gid >= batchSize) return; // Threads explicitly indexed per sample

    float mean = savedMean[gid];
    float variance = savedVariance[gid];
    float invStd = rsqrt(variance + epsilon);

    // Intermediate sums for gamma/beta gradients (per-sample)
    float grad_gamma_local[1024]; // adjust if featureDim > 1024
    float grad_beta_local[1024];  // adjust if featureDim > 1024

    // Initialize accumulators
    for (int f = 0; f < featureDim; f++) {
        grad_gamma_local[f] = 0.0f;
        grad_beta_local[f] = 0.0f;
    }

    // Compute gradients explicitly for gamma and beta per-sample
    for (int f = 0; f < featureDim; f++) {
        int idx = gid * featureDim + f;

        float xhat = (input[idx] - mean) * invStd;
        float dY = inputErrors[idx];

        grad_gamma_local[f] = dY * xhat;
        grad_beta_local[f] = dY;
    }

    // Reduce gradients explicitly and update gamma and beta
    for (int f = 0; f < featureDim; f++) {
        // Atomic updates (if multiple threads share gamma/beta across samples)
        atomic_fetch_add_explicit((device atomic_float*)&gamma[f], -learningRate * grad_gamma_local[f] / float(batchSize), memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&beta[f], -learningRate * grad_beta_local[f] / float(batchSize), memory_order_relaxed);
    }

    // Compute input gradient explicitly (outputErrors)
    float sum_dY = 0.0f;
    float sum_dY_xhat = 0.0f;

    // Compute sum_dY and sum_dY_xhat explicitly
    for (int f = 0; f < featureDim; f++) {
        int idx = gid * featureDim + f;
        float xhat = (input[idx] - mean) * invStd;
        float dY = inputErrors[idx];
        sum_dY += dY;
        sum_dY_xhat += dY * xhat;
    }

    // Compute final output errors explicitly
    for (int f = 0; f < featureDim; f++) {
        int idx = gid * featureDim + f;
        float xhat = (input[idx] - mean) * invStd;
        float dY = inputErrors[idx];

        float dX = (gamma[f] * invStd / float(featureDim)) *
                   (float(featureDim) * dY - sum_dY - xhat * sum_dY_xhat);

        outputErrors[idx] = dX;
    }
}
