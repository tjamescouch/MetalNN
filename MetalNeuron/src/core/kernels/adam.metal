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


kernel void adam_kernel(
    device float* parameters         [[buffer(0)]],   // Parameters (weights or biases)
    device float* gradients          [[buffer(1)]],   // Gradients accumulated across the batch
    device float* m                  [[buffer(2)]],   // First moment vector
    device float* v                  [[buffer(3)]],   // Second moment vector
    constant float& learning_rate    [[buffer(4)]],   // Base learning rate
    constant float& beta1            [[buffer(5)]],   // Exponential decay rate for first moment
    constant float& beta2            [[buffer(6)]],   // Exponential decay rate for second moment
    constant float& epsilon          [[buffer(7)]],   // Prevent division by zero
    constant uint& batch_size        [[buffer(8)]],   // Current batch size
    constant uint& timestep          [[buffer(9)]],   // Global step for bias correction
    constant uint& param_count       [[buffer(10)]],  // Total number of parameters
    uint tid                         [[thread_position_in_grid]]
)
{
    // Each thread handles one parameter index
    if (tid >= param_count) return;

    // 1) Average the gradient over the batch (assuming 'gradients' is the sum)
    float grad_avg = gradients[tid] / (float)batch_size;

    // 2) Update biased first moment estimate (m) and second moment estimate (v)
    m[tid] = beta1 * m[tid] + (1.0f - beta1) * grad_avg;
    v[tid] = beta2 * v[tid] + (1.0f - beta2) * grad_avg * grad_avg;

    // 3) Skip explicit bias correction; keep old m & v directly.
    float m_hat = m[tid];
    float v_hat = v[tid];

    // 4) Compute the Adam update (no bias correction)
    float update = learning_rate * (m_hat / (sqrt(v_hat) + epsilon));

    // Optional clamp to prevent extreme updates
    update = clamp(update, -1e3f, 1e3f);

    // 5) Apply the update
    parameters[tid] -= update;

    // 6) Reset gradient accumulator for next batch
    gradients[tid] = 0.0f;
}

