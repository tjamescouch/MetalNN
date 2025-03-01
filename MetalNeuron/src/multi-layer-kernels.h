#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Global constants
constant float learning_rate_w = 0.00005f;
constant float learning_rate_b = 0.00005f;

// Activation function and derivative for hidden layers
inline float hiddenActivation(float x) {
  return tanh(x);
}

inline float hiddenActivationDerivative(float y) {
  // derivative = 1 - y^2 (for tanh)
  return clamp(1.0f - y * y, -0.5, 0.5);
}

// Activation function and derivative for output layer (linear)
inline float outputActivation(float x) {
  return x;  // Linear activation
}

inline float outputActivationDerivative(float y) {
  return 1.0f;  // Derivative of linear activation is always 1
}


inline float random(uint seed) {
    seed = seed * 1664525 + 1013904223;
    return float(seed & 0x00FFFFFF) / float(0x01000000);
}

//-------------------------------------------------------------------
// Forward pass for the recurrent layer (RNN cell)
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
    
    // Contribution from current input
    for (uint i = 0; i < input_dim; i++) {
        sum += x[i] * W_xh[i * hidden_dim + tid];
    }
    
    // Recurrent contribution from previous hidden state
    for (uint j = 0; j < hidden_dim; j++) {
        sum += h_prev[j] * W_hh[j * hidden_dim + tid];
    }
    
    h[tid] = hiddenActivation(clamp(sum, -20.f, 20.f));
}

//-------------------------------------------------------------------
// Forward pass for the output layer (feedforward)
kernel void forward_output_layer(
    device const float* h            [[buffer(0)]],
    device       float* y            [[buffer(1)]],
    device const float* W            [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device const uint* pH            [[buffer(4)]],
    device const uint* pN            [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;

    if (tid >= output_dim) return;
    
    float sum = b[tid];
    for (uint i = 0; i < hidden_dim; i++) {
        sum += h[i] * W[i * output_dim + tid];
    }
    y[tid] = outputActivation(sum);  // linear activation, no clamping needed
}

//-------------------------------------------------------------------
// Learning kernel for the output layer
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

    float raw_error = y[tid] - y_hat[tid];
    float delta = raw_error * outputActivationDerivative(y[tid]); 
    delta = clamp(delta, -1.0f, 1.0f);
    error[tid] = delta;

    // Weight + bias update
    for (uint i = 0; i < hidden_dim; i++) {
        W[i * output_dim + tid] -= learning_rate_w * delta * h[i];
    }
    b[tid] -= learning_rate_b * delta;
}

//-------------------------------------------------------------------
// Learning kernel for the recurrent layer (multi-step BPTT)
// CHANGED: added next_hidden_error
kernel void learn_rnn(
    device const float* x            [[buffer(0)]],
    device const float* h_prev       [[buffer(1)]],
    device       float* W_xh         [[buffer(2)]],
    device       float* W_hh         [[buffer(3)]],
    device       float* b            [[buffer(4)]],
    device const float* h            [[buffer(5)]],
    device const float* output_error [[buffer(6)]],
    device const float* next_hidden_error [[buffer(7)]], 
    device       float* hidden_error [[buffer(8)]],
    device const uint* pX            [[buffer(9)]],
    device const uint* pH            [[buffer(10)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;

    if (tid >= hidden_dim) return;

    // Combine the next timestep's hidden error plus local output_error
    float accumulated_err = output_error[tid];
    for (uint k = 0; k < hidden_dim; k++) {
        accumulated_err += next_hidden_error[k] * W_hh[k * hidden_dim + tid];
    }

    // Multiply by activation derivative of current hidden state
    float delta = accumulated_err * hiddenActivationDerivative(h[tid]);
    delta = clamp(delta, -1.0f, 1.0f);
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

)"; // end of the big string

} // namespace multilayerkernels

#pragma endregion Declarations }
#endif
