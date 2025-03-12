#include <metal_stdlib>
using namespace metal;

#include "common.metal"

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

constant float threshold    = 1.0f;
constant float decay_factor = 1.0f;


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);
//-------------------------------------------------------------------
// Forward pass for the recurrent layer (RNN cell)
kernel void forward_rnn(
    device const float* x            [[buffer(0)]],
    device       float* h_prev       [[buffer(1)]],
    device       float* h            [[buffer(2)]],
    device const float* W_xh         [[buffer(3)]],
    device const float* W_hh         [[buffer(4)]],
    device const float* b            [[buffer(5)]],
    device const uint* pX            [[buffer(6)]],
    device const uint* pH            [[buffer(7)]],
    device const uint* activation    [[buffer(8)]],
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
    
    h_prev[tid] = h[tid];
    h[tid] = activate(sum, *activation);
}

//-------------------------------------------------------------------
// Learning kernel for the recurrent layer (multi-step BPTT)
//-------------------------------------------------------------------
kernel void learn_rnn(
    device const float* x            [[buffer(0)]],
    device const float* h_prev       [[buffer(1)]],
    device       float* W_xh         [[buffer(2)]],
    device       float* W_hh         [[buffer(3)]],
    device       float* b            [[buffer(4)]],
    device const float* h            [[buffer(5)]],
    device       float* output_error [[buffer(6)]],
    device const float* next_hidden_error [[buffer(7)]],
    device       float* hidden_error [[buffer(8)]],
    device const uint* pX            [[buffer(9)]],
    device const uint* pH            [[buffer(10)]],
    device       float* pDecay       [[buffer(11)]],
    device const uint* activation    [[buffer(12)]],
    constant     uint& batch_size    [[buffer(13)]],
    constant     float& learning_rate [[buffer(14)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;

    if (tid >= hidden_dim) return;

    if (tid == 0) {
        *pDecay *= decay_factor;
    }
    float decay = *pDecay;


    // Combine the next timestep's hidden error plus local output_error
    float accumulated_err = output_error[tid];
    for (uint k = 0; k < hidden_dim; k++) {
        accumulated_err += next_hidden_error[k] * W_hh[k * hidden_dim + tid];
    }

    // Multiply by activation derivative of current hidden state
    float delta = accumulated_err * activate_derivative(h[tid], *activation);
    delta = clamp(delta, -threshold, threshold);
    hidden_error[tid] = delta;
    output_error[tid] = delta;

    // Update input-to-hidden weights
    for (uint i = 0; i < input_dim; i++) {
        W_xh[i * hidden_dim + tid] -= learning_rate * delta * x[i] * decay;
    }

    // Update recurrent weights
    for (uint j = 0; j < hidden_dim; j++) {
        W_hh[j * hidden_dim + tid] -= learning_rate * delta * h_prev[j] * decay;
    }

    // Update bias
    b[tid] -= learning_rate * delta * decay;
}
