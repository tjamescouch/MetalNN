// multiLayerKernels.h
#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Global constants
constant float learning_rate_w = 0.001f;
constant float learning_rate_b = 0.001f;
constant float min_delta      = 0.1f;
constant float max_de_dw      = 0.5f;
constant float max_de_db      = 0.1f;

// Activation function and its derivative
inline float activationFunction(float x) {
  return tanh(x);
}

inline float activationDerivative(float y) {
  // Using y = tanh(x): derivative = 1 - y^2
  return 1.0f - y * y;
}

//
// Forward pass for a single layer
//   - input: activation vector from previous layer
//   - output: activation vector for current layer
//   - W: weight matrix of shape [M x N] (M: number of inputs, N: neurons)
//   - b: bias vector for the current layer
//   - pM: pointer to M, pN: pointer to N
//
kernel void forward_layer(
    device const float* input        [[buffer(0)]],
    device       float* output       [[buffer(1)]],
    device const float* W            [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device const uint* pM            [[buffer(4)]],
    device const uint* pN            [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint M = *pM;
    uint N = *pN;

    if (tid >= N) return;

    float sum = b[tid];
    for (uint i = 0; i < M; i++) {
        sum += input[i] * W[i * N + tid];
    }
    output[tid] = activationFunction(clamp(sum, -10.f, 10.f));
}

//
// Learning kernel for the output layer
//   - Computes the error and accumulates weight and bias updates.
//   - y: actual output; y_hat: target output.
kernel void learn_output_layer(
    device const float* x            [[buffer(0)]],
    device       float* W            [[buffer(1)]],
    device       float* b            [[buffer(2)]],
    device       float* y            [[buffer(3)]],
    device const float* y_hat        [[buffer(4)]],
    device       float* error        [[buffer(5)]],
    device const uint* pM            [[buffer(6)]],
    device const uint* pN            [[buffer(7)]],
    device       float* W_accumulator[[buffer(8)]],
    device       float* b_accumulator[[buffer(9)]],
    uint tid                         [[thread_position_in_grid]]
) {

    uint M = *pM;
    uint N = *pN;
    if (tid >= N) return;

    // Compute weight updates
    float delta_w, abs_delta_w, delta_error;
    for (uint i = 0; i < M; i++) {
        delta_w = W[i * N + tid] - W_accumulator[i * N + tid];
        abs_delta_w = fabs(delta_w);
        float y_hat_minus_y = y_hat[tid] - y[tid];
        error[tid] = y_hat_minus_y * y_hat_minus_y * y_hat_minus_y;
        delta_error = error[tid];
        
        float delta_w_no_zero = abs_delta_w > min_delta ? delta_w : sign(delta_w) * min_delta;
        float de_dw = clamp(delta_error / delta_w_no_zero, -max_de_dw, max_de_dw);
        
        W_accumulator[i * N + tid] += learning_rate_w * error[tid] * x[i] * de_dw * sign(de_dw);
        W_accumulator[i * N + tid] = clamp(W_accumulator[i * N + tid], -0.1f, 0.1f);
    }

    // Compute bias updates
    float delta_b = b[tid] - b_accumulator[tid];
    float abs_delta_b = fabs(delta_b);
    float delta_b_no_zero = abs_delta_b > min_delta ? delta_b : sign(delta_b) * min_delta;
    float de_db = clamp(error[tid] / delta_b_no_zero, -max_de_db, max_de_db);

    b_accumulator[tid] -= clamp(learning_rate_b * error[tid] * de_db * sign(de_db), -0.1f, 0.1f);
}

//
// Learning kernel for a hidden layer
//   - Propagates error from the next layer back into the current layer.
//   - error_next: error from the layer ahead; W_next: weights from the current layer to next layer.
kernel void learn_hidden_layer(
    device const float* x             [[buffer(0)]],  // activation from previous layer
    device       float* W             [[buffer(1)]],
    device       float* b             [[buffer(2)]],
    device       float* y             [[buffer(3)]],  // current layer activation
    device       float* error         [[buffer(4)]],
    device const float* error_next    [[buffer(5)]],
    device const float* W_next        [[buffer(6)]],
    device const uint* pM             [[buffer(7)]],  // input size for current layer
    device const uint* pN             [[buffer(8)]],  // number of neurons in current layer
    device const uint* pN_next        [[buffer(9)]],  // number of neurons in next layer
    device       float* W_accumulator [[buffer(10)]],
    device       float* b_accumulator [[buffer(11)]],
    uint tid                        [[thread_position_in_grid]]
) {
    uint M = *pM;
    uint N = *pN;
    uint N_next = *pN_next;
    if (tid >= N) return;
    
    // Backpropagate the error from the next layer:
    float error_sum = 0.0f;
    for (uint j = 0; j < N_next; j++) {
        // For current neuron 'tid', weight connecting it to neuron 'j' in next layer:
        error_sum += W_next[tid * N_next + j] * error_next[j];
    }
    error[tid] = error_sum;

    // Compute weight updates
    float delta_w, abs_delta_w, delta_error;
    for (uint i = 0; i < M; i++) {
        delta_w = W[i * N + tid] - W_accumulator[i * N + tid];
        abs_delta_w = fabs(delta_w);
        delta_error = error[tid];
        
        float delta_w_no_zero = abs_delta_w > min_delta ? delta_w : sign(delta_w) * min_delta;
        float de_dw = clamp(delta_error / delta_w_no_zero, -max_de_dw, max_de_dw);
        
        W_accumulator[i * N + tid] += learning_rate_w * error[tid] * x[i] * de_dw * sign(de_dw);
        W_accumulator[i * N + tid] = clamp(W_accumulator[i * N + tid], -0.1f, 0.1f);
    }

    // Compute bias updates
    float delta_b = b[tid] - b_accumulator[tid];
    float abs_delta_b = fabs(delta_b);
    float delta_b_no_zero = abs_delta_b > min_delta ? delta_b : sign(delta_b) * min_delta;
    float de_db = clamp(error[tid] / delta_b_no_zero, -max_de_db, max_de_db);

    b_accumulator[tid] -= clamp(learning_rate_b * error[tid] * de_db * sign(de_db), -0.1f, 0.1f);
}

//
// Apply updates kernel (can be used for each layer)
//   - Applies accumulated weight and bias updates to the network parameters.
kernel void apply_updates(
    device       float* W             [[buffer(0)]],
    device       float* b             [[buffer(1)]],
    device       float* W_accumulator [[buffer(2)]],
    device       float* b_accumulator [[buffer(3)]],
    device const uint* pM            [[buffer(4)]],
    device const uint* pN            [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint M = *pM;
    uint N = *pN;
    if (tid >= N) return;
    
    // Update each weight for the current neuron
    for (uint i = 0; i < M; i++) {
        W[i * N + tid] += W_accumulator[i * N + tid];
        W_accumulator[i * N + tid] = 0.0f;
    }
    
    // Update bias and reset accumulator
    b[tid] += b_accumulator[tid];
    b[tid] = clamp(b[tid], -1.0f, 1.0f);
    b_accumulator[tid] = 0.0f;
}
)";

} // namespace kernels

#pragma endregion Declarations }
#endif
