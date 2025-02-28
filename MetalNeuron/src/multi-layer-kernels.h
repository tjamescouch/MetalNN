// multiLayerKernels.h
#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Global constants
constant float learning_rate_w = 0.05f;
constant float learning_rate_b = 0.01f;

// Activation function and its derivative
inline float activationFunction(float x) {
  return tanh(x);
}

inline float activationDerivative(float y) {
  // Using y = tanh(x): derivative = 1 - y^2
  return (1.0f - y * y);
}


inline float random(uint seed) {
    seed = seed * 1664525 + 1013904223;
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

//-------------------------------------------------------------------
// Forward pass for the recurrent layer (RNN cell)
// Computes the hidden state using the current input and the previous hidden state.
kernel void forward_rnn(
    device const float* x            [[buffer(0)]],
    device const float* h_prev       [[buffer(1)]],
    device       float* h            [[buffer(2)]],
    device const float* W_xh         [[buffer(3)]],
    device const float* W_hh         [[buffer(4)]],
    device const float* b            [[buffer(5)]],
    device const uint* pX            [[buffer(6)]],
    device const uint* pH            [[buffer(7)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;
    
    if (tid >= hidden_dim) return;
    
    float sum = b[tid];
    
    // Contribution from current input: x * W_xh
    for (uint i = 0; i < input_dim; i++) {
        sum += x[i] * W_xh[i * hidden_dim + tid];
    }
    
    // Recurrent contribution from previous hidden state: h_prev * W_hh
    for (uint j = 0; j < hidden_dim; j++) {
        sum += h_prev[j] * W_hh[j * hidden_dim + tid];
    }
    
    h[tid] = activationFunction(clamp(sum, -20.f, 20.f));
}

//-------------------------------------------------------------------
// Forward pass for the output layer (feedforward)
// Computes the output activation from the hidden state produced by the RNN.
kernel void forward_output_layer(
    device const float* h            [[buffer(0)]],  // hidden state from RNN layer
    device       float* y            [[buffer(1)]],  // output activation
    device const float* W            [[buffer(2)]],  // weight matrix (hidden_dim x output_dim)
    device const float* b            [[buffer(3)]],  // bias vector for output layer
    device const uint* pH            [[buffer(4)]],  // hidden state dimension
    device const uint* pN            [[buffer(5)]],  // number of output neurons
    uint tid                         [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;
    
    if (tid >= output_dim) return;
    
    float sum = b[tid];
    for (uint i = 0; i < hidden_dim; i++) {
        sum += h[i] * W[i * output_dim + tid];
    }
    y[tid] = activationFunction(clamp(sum, -20.f, 20.f));
}

//-------------------------------------------------------------------
// Learning kernel for the recurrent layer (RNN cell)
// Updates both the input-to-hidden and recurrent weight matrices based on the error signal.
kernel void learn_output_layer(
    device const float* h            [[buffer(0)]],
    device       float* W            [[buffer(1)]],
    device       float* b            [[buffer(2)]],
    device       float* y            [[buffer(3)]],
    device const float* y_hat        [[buffer(4)]],
    device       float* error        [[buffer(5)]],
    device const uint* pH            [[buffer(6)]],
    device const uint* pN            [[buffer(7)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;
    
    if (tid >= output_dim) return;

    // Compute raw error
    float raw_error = y[tid] - y_hat[tid];

    float delta = raw_error * activationDerivative(y[tid]);
    error[tid] = delta;

    // Update weights and biases
    for (uint i = 0; i < hidden_dim; i++) {
        W[i * output_dim + tid] -= learning_rate_w * delta * h[i];
    }
    b[tid] -= learning_rate_b * delta;
}

//-------------------------------------------------------------------
// Learning kernel for the recurrent layer (RNN cell)
kernel void learn_rnn(
    device const float* x            [[buffer(0)]],
    device const float* h_prev       [[buffer(1)]],
    device       float* W_xh         [[buffer(2)]],
    device       float* W_hh         [[buffer(3)]],
    device       float* b            [[buffer(4)]],
    device const float* h            [[buffer(5)]],
    device const float* output_error [[buffer(6)]], // error from next layer
    device       float* hidden_error [[buffer(7)]], // hidden layer error (to propagate backward)
    device const uint* pX            [[buffer(8)]],
    device const uint* pH            [[buffer(9)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;

    if (tid >= hidden_dim) return;

    // Compute propagated hidden error using activation derivative
    float propagated_error = 0.0f;

    // Corrected: Propagate error using the transpose of W_hh
    for (uint k = 0; k < hidden_dim; k++) {
        propagated_error += output_error[k] * W_hh[k * hidden_dim + tid];
    }

    // Multiply by activation derivative
    float delta = propagated_error * activationDerivative(h[tid]);
    hidden_error[tid] = delta;

    // Update input-to-hidden weights
    for (uint i = 0; i < input_dim; i++) {
        W_xh[i * hidden_dim + tid] -= learning_rate_w * delta * x[i];
    }

    // Update recurrent weights
    for (uint j = 0; j < hidden_dim; j++) {
        W_hh[j * hidden_dim + tid] -= learning_rate_w * delta * h_prev[j];
    }

    // Update bias
    b[tid] -= learning_rate_b * delta;
}

)";

} // namespace kernels

#pragma endregion Declarations }
#endif
