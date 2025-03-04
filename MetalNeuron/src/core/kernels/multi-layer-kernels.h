#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

inline void softmax(const device float* input, device float* output, uint outputDim) {
    float maxVal = input[0];
    for (uint i = 1; i < outputDim; ++i) {
        maxVal = max(maxVal, input[i]);
    }

    float sumExp = 0.0f;
    for (uint i = 0; i < outputDim; ++i) {
        sumExp += exp(input[i] - maxVal);
    }

    for (uint i = 0; i < outputDim; ++i) {
        output[i] = exp(input[i] - maxVal) / sumExp;
    }
}

// Global constants
constant float learning_rate_w = 0.01f;
constant float learning_rate_b = 0.01f;
constant float decay_factor = 0.99999f;
constant float threshold = 0.1f;

// Activation functions
inline float activate(const float x, const uint activation) {
    switch (activation) {
        case 0: return x;                      // Linear
        case 1: return max(0.0f, x);           // ReLU
        case 2: return tanh(x);                // Tanh
        case 3: return 1.0f / (1.0f + exp(-x)); // Sigmoid
        default: return 0.0f;                   // Error return 0
    }
}

// Activation derivatives
inline float activate_derivative(const float y, const uint activation) {
    switch (activation) {
        case 0: return 1.0f;                   // Linear
        case 1: return y > 0.0f ? 1.0f : 0.0f; // ReLU
        case 2: return 1.0f - y * y;           // Tanh
        case 3: return y * (1.0f - y);         // Sigmoid
        default: return 0.0f;                  // Error return 0
    }
}

kernel void forward_dense_layer(
    device const float* h            [[buffer(0)]],
    device       float* y            [[buffer(1)]],
    device const float* W            [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device const uint* pH            [[buffer(4)]],
    device const uint* pN            [[buffer(5)]],
    device const uint* activation    [[buffer(6)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;

    if (tid >= output_dim) return;

    float sum = b[tid];
    for (uint i = 0; i < hidden_dim; i++) {
        sum += h[i] * W[i * output_dim + tid];
    }

    // First compute the pre-activation sums.
    y[tid] = sum;

    // Handle softmax separately after all sums are computed.
    if (*activation == 4) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) { // Single thread computes softmax for stability
            softmax(y, y, output_dim);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        y[tid] = activate(sum, *activation);  // Use existing activate for other activations
    }
}


kernel void learn_dense_layer(
    device const float* h              [[buffer(0)]],
    device float* W                    [[buffer(1)]],
    device float* b                    [[buffer(2)]],
    device const float* y_hat          [[buffer(3)]],
    device const float* y              [[buffer(4)]],
    device float* error                [[buffer(5)]],
    device const uint* pH              [[buffer(6)]],
    device const uint* pN              [[buffer(7)]],
    device float* pDecay               [[buffer(8)]],
    device const uint* activation      [[buffer(9)]],
    device float* debug                [[buffer(10)]],
    device float* prevLayerErrors      [[buffer(11)]], // explicitly propagate errors backward
    uint tid                           [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;

    if (tid >= output_dim) return;

    if (tid == 0) {
        *pDecay *= decay_factor;
    }
    float decay = *pDecay;

    float raw_error = y_hat[tid] - y[tid];
    debug[tid] = raw_error;

    float delta = raw_error * activate_derivative(y_hat[tid], *activation);
    delta = clamp(delta, -threshold, threshold);
    error[tid] = delta;

    for (uint i = 0; i < hidden_dim; i++) {
        prevLayerErrors[i] += W[i * output_dim + tid] * delta;
        W[i * output_dim + tid] -= learning_rate_w * delta * h[i] * decay;
    }

    b[tid] -= learning_rate_b * delta * decay;
}

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
    device const float* output_error [[buffer(6)]],
    device const float* next_hidden_error [[buffer(7)]], 
    device       float* hidden_error [[buffer(8)]],
    device const uint* pX            [[buffer(9)]],
    device const uint* pH            [[buffer(10)]],
    device       float* pDecay       [[buffer(11)]],
    device const uint* activation    [[buffer(12)]],
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

    // Update input-to-hidden weights
    for (uint i = 0; i < input_dim; i++) {
        W_xh[i * hidden_dim + tid] -= learning_rate_w * delta * x[i] * decay;
    }

    // Update recurrent weights
    for (uint j = 0; j < hidden_dim; j++) {
        W_hh[j * hidden_dim + tid] -= learning_rate_w * delta * h_prev[j] * decay;
    }

    // Update bias
    b[tid] -= learning_rate_b * delta * decay;
}

//-------------------------------------------------------------------
// Forward pass for Dropout layer (CPU-generated randomness)
kernel void forward_dropout(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const float* randomMask  [[buffer(2)]],
    device const float* rate        [[buffer(3)]],
    device const uint* featureDim   [[buffer(4)]],
    constant bool& isTraining       [[buffer(5)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid >= *featureDim) return;

    if (isTraining) {
        output[tid] = (1.0f - *rate) * input[tid];
    } else {
        output[tid] = randomMask[tid] >= *rate ? input[tid] : 0;
    }
}

//-------------------------------------------------------------------
// Backward pass for Dropout layer
kernel void backward_dropout(
    device const float* output_error [[buffer(0)]],
    device float* input_error        [[buffer(1)]],
    device const float* randomMask   [[buffer(2)]],
    device const float* rate         [[buffer(3)]],
    device const uint* featureDim    [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= *featureDim) return;

    input_error[tid] = randomMask[tid] >= *rate ? output_error[tid] : 0;
}

kernel void forward_batch_norm(
    device const float* input         [[buffer(0)]],
    device float* output              [[buffer(1)]],
    device float* gamma               [[buffer(2)]],
    device float* beta                [[buffer(3)]],
    device float* runningMean         [[buffer(4)]],
    device float* runningVariance     [[buffer(5)]],
    constant float& epsilon           [[buffer(6)]],
    constant int& featureDim          [[buffer(7)]],
    constant bool& isTraining         [[buffer(8)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= (uint)featureDim) return;

    float mean = runningMean[tid];
    float variance = runningVariance[tid];

    float normalized = (input[tid] - mean) / sqrt(variance + epsilon);
    output[tid] = gamma[tid] * normalized + beta[tid];
}

kernel void backward_batch_norm(
    device const float* output        [[buffer(0)]],
    device const float* outputError   [[buffer(1)]],
    device float* inputError          [[buffer(2)]],
    device float* gamma               [[buffer(3)]],
    device float* beta                [[buffer(4)]],
    constant float& epsilon           [[buffer(5)]],
    constant int& featureDim          [[buffer(6)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= (uint)featureDim) return;

    // Simplified initial gradient propagation (full implementation requires batch stats)
    float normalized = (output[tid] - beta[tid]) / (gamma[tid] + epsilon);
    inputError[tid] = outputError[tid] * gamma[tid] / sqrt(normalized + epsilon);

    // Simple parameter updates (extendable for optimization algorithms)
    gamma[tid] -= 0.001f * outputError[tid] * normalized;
    beta[tid] -= 0.001f * outputError[tid];
}

// The Adam optimizer kernel
kernel void adam_kernel(device float* params          [[buffer(0)]],
                        device const float* grads     [[buffer(1)]],
                        device float* m               [[buffer(2)]],
                        device float* v               [[buffer(3)]],
                        constant uint& timestep       [[buffer(4)]],
                        constant float& learning_rate [[buffer(5)]],
                        constant float& beta1         [[buffer(6)]],
                        constant float& beta2         [[buffer(7)]],
                        constant float& epsilon       [[buffer(8)]],
                        uint index                    [[thread_position_in_grid]])
{
    float grad = grads[index];

    // Update first moment estimate
    m[index] = beta1 * m[index] + (1.0 - beta1) * grad;

    // Update second moment estimate
    v[index] = beta2 * v[index] + (1.0 - beta2) * grad * grad;

    // Compute bias-corrected estimates
    float m_hat = m[index] / (1.0 - pow(beta1, timestep));
    float v_hat = v[index] / (1.0 - pow(beta2, timestep));

    // Update parameter
    params[index] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

)";

} // namespace multilayerkernels

#pragma endregion Declarations }
#endif
