#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
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


// Global constants
constant float decay_factor = 1.0f;//0.9999999f;
constant float threshold = 0.1f;
constant float max_abs_sum = 1000.0f;

// Activation functions
inline float activate(const float x, const uint activation) {
    switch (activation) {
        case ACTIVATION_LINEAR: return x;                      // Linear
        case ACTIVATION_RELU: return max(0.0f, x);           // ReLU
        case ACTIVATION_TANH: return tanh(x);                // Tanh
        case ACTIVATION_SIGMOID: return 1.0f / (1.0f + exp(-x)); // Sigmoid
        default: return 0.0f;                   // Error return 0
    }
}

// Activation derivatives
inline float activate_derivative(const float y, const uint activation) {
    switch (activation) {
        case ACTIVATION_LINEAR: return 1.0f;                   // Linear
        case ACTIVATION_RELU: return y > 0.0f ? 1.0f : 0.0f; // ReLU
        case ACTIVATION_TANH: return 1.0f - y * y;           // Tanh
        case ACTIVATION_SIGMOID: return y * (1.0f - y);         // Sigmoid
        default: return 0.0f;                  // Error return 0
    }
}

kernel void forward_dense_layer(
    device const float* h         [[buffer(0)]],
    device       float* y         [[buffer(1)]],
    device const float* W         [[buffer(2)]],
    device const float* b         [[buffer(3)]],
    device const uint* pH         [[buffer(4)]],
    device const uint* pN         [[buffer(5)]],
    device const uint* activation [[buffer(6)]],
    constant uint& batchSize      [[buffer(7)]],
    device float* debug           [[buffer(8)]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint hidden_dim = *pH;
    uint output_dim = *pN;

    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;
    if (sample_id >= batchSize || neuron_id >= output_dim) return;

    threadgroup float shared_y[1024]; 

    float sum = b[neuron_id];
    for (uint i = 0; i < hidden_dim; ++i) {
        sum += h[sample_id * hidden_dim + i] * W[i * output_dim + neuron_id];
    }

    shared_y[tid] = clamp(sum, -max_abs_sum, max_abs_sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (*activation == ACTIVATION_SOFTMAX) {
        uint sample_offset = sample_id * output_dim;

        // Numerically stable softmax per sample
        float maxVal = shared_y[sample_offset];
        for (uint i = 1; i < output_dim; ++i)
            maxVal = max(maxVal, shared_y[sample_offset + i]);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_y[tid] = exp(shared_y[tid] - maxVal);
        

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (neuron_id == 0) {
            float sumExp = 0.0f;
            for (uint i = 0; i < output_dim; ++i)
                sumExp += shared_y[sample_offset + i];

            for (uint i = 0; i < output_dim; ++i)
                shared_y[sample_offset + i] /= sumExp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        debug[tid] = shared_y[tid];
    } else {
        shared_y[tid] = activate(shared_y[tid], *activation);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    y[gid] = shared_y[tid];
}

kernel void learn_non_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // input activations
    device float* W                      [[buffer(1)]],  // weights
    device const float* b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // predicted outputs
    device const float* inputErrors      [[buffer(4)]],  // errors from next layer
    device float* error                  [[buffer(5)]],  // output error (delta)
    constant uint& input_dim             [[buffer(6)]],  // input dimension
    constant uint& output_dim            [[buffer(7)]],  // output dimension
    device float* pDecay                 [[buffer(8)]],  // decay factor
    constant uint& activation            [[buffer(9)]],  // activation type
    device float* debug                  [[buffer(10)]], // debug buffer
    device float* prevLayerErrors        [[buffer(11)]], // errors to prev layer
    constant uint& batch_size            [[buffer(12)]], // batch size
    device const float* pLearningRate    [[buffer(13)]], // learning rate
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay = *pDecay;
    float learning_rate = *pLearningRate;

    // Compute index offsets based on batch sample
    const device float* sample_h = h + (sample_id * input_dim);
    const device float* sample_y_hat = y_hat + (sample_id * output_dim);
    const device float* sample_outputErrors = inputErrors + (sample_id * output_dim);
    device float* sample_error = error + (sample_id * output_dim);
    device float* sample_prevLayerErrors = prevLayerErrors + (sample_id * input_dim);

    float raw_error = sample_outputErrors[neuron_id];
    debug[gid] = raw_error;

    float delta = raw_error * activate_derivative(sample_y_hat[neuron_id], activation);

    delta = clamp(delta, -threshold, threshold);
    sample_error[neuron_id] = delta;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate weight gradients for each input neuron
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;
        grad = clamp(grad, -threshold, threshold);

        // Atomic add for concurrent accumulation across batches
        atomic_fetch_add_explicit((device atomic_float*)&W[i * output_dim + neuron_id], -learning_rate * grad * decay, memory_order_relaxed);

        // Accumulate error signals for previous layer (for backprop)
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevLayerErrors[i], W[i * output_dim + neuron_id] * delta, memory_order_relaxed);
    }

    // Update biases atomically
    atomic_fetch_add_explicit((device atomic_float*)&b[neuron_id], -learning_rate * delta * decay, memory_order_relaxed);

}


kernel void learn_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // input activations
    device float* W                      [[buffer(1)]],  // weights
    device float* b                      [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // predicted outputs
    device const float* y                [[buffer(4)]],  // targets
    device float* error                  [[buffer(5)]],  // output error (delta)
    constant uint& input_dim             [[buffer(6)]],  // input dimension
    constant uint& output_dim            [[buffer(7)]],  // output dimension
    device float* pDecay                 [[buffer(8)]],  // decay factor
    constant uint& activation            [[buffer(9)]],  // activation type
    device float* debug                  [[buffer(10)]], // debug buffer
    device float* prevLayerErrors        [[buffer(11)]], // errors to prev layer
    constant uint& batch_size            [[buffer(12)]], // batch size
    device const float* pLearningRate    [[buffer(13)]], // learning rate
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay = *pDecay;
    float learning_rate = *pLearningRate;

    // Compute index offsets based on batch sample
    const device float* sample_h = h + (sample_id * input_dim);
    const device float* sample_y_hat = y_hat + (sample_id * output_dim);
    const device float* sample_y = y + (sample_id * output_dim);
    device float* sample_error = error + (sample_id * output_dim);
    device float* sample_prevLayerErrors = prevLayerErrors + (sample_id * input_dim);

    float raw_error = sample_y_hat[neuron_id] - sample_y[neuron_id];
    debug[gid] = raw_error;

    

    float delta;
    if (activation == ACTIVATION_SOFTMAX) { // softmax
        delta = raw_error;
    } else {
        delta = raw_error * activate_derivative(sample_y_hat[neuron_id], activation);
    }

    delta = clamp(delta, -threshold, threshold);
    sample_error[neuron_id] = delta;


    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate weight gradients for each input neuron
    for (uint i = 0; i < input_dim; i++) {
                
        float grad = sample_h[i] * delta;

        grad = clamp(grad, -threshold, threshold);

        // Atomic add for concurrent accumulation across batches
        atomic_fetch_add_explicit((device atomic_float*)&W[i * output_dim + neuron_id], -learning_rate * grad * decay, memory_order_relaxed);

        // Accumulate error signals for previous layer (for backprop)
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevLayerErrors[i], W[i * output_dim + neuron_id] * delta, memory_order_relaxed);
    }

    // Update biases atomically
    atomic_fetch_add_explicit((device atomic_float*)&b[neuron_id], -learning_rate * delta * decay, memory_order_relaxed);
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
    device const float* input_error [[buffer(0)]],
    device float* output_error        [[buffer(1)]],
    device const float* randomMask   [[buffer(2)]],
    device const float* rate         [[buffer(3)]],
    device const uint* featureDim    [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= *featureDim) return;

    output_error[tid] = randomMask[tid] >= *rate ? input_error[tid] : 0;
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

kernel void adam_kernel(
    device float* parameters        [[buffer(0)]],   // Parameters (weights or biases)
    device float* gradients          [[buffer(1)]],  // Gradients accumulated
    device float* m                  [[buffer(2)]],  // First moment vector
    device float* v                  [[buffer(3)]],  // Second moment vector
    constant float& learning_rate    [[buffer(4)]],  // Learning rate
    constant float& beta1            [[buffer(5)]],  // Exponential decay rate for first moment estimates
    constant float& beta2            [[buffer(6)]],  // Second moment vector exponential decay rate
    constant float& epsilon          [[buffer(7)]],  // Stability factor to prevent division by zero
    constant uint& batch_size        [[buffer(8)]],  // Current batch size
    constant uint& timestep          [[buffer(9)]],  // Timestep for bias correction
    constant uint& param_count       [[buffer(10)]], // Total number of parameters
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= param_count) return;

    // Compute average gradient across batch
    float grad_avg = gradients[tid] / float(batch_size);

    // Adam moment estimates
    m[tid] = beta1 * m[tid] + (1.0f - beta1) * grad_avg;
    v[tid] = beta2 * v[tid] + (1.0f - beta2) * grad_avg * grad_avg;

    // Bias-corrected first and second moment estimates
    float m_hat = m[tid] / (1.0f - pow(beta1, timestep));
    float v_hat = v[tid] / (1.0f - pow(beta2, timestep));

    // Parameter update with stability clamp
    float update = learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    update = clamp(update, -1e3f, 1e3f); // Aggressive clamp to prevent explosions

    parameters[tid] -= update;

    // Reset gradient accumulator for next batch
    gradients[tid] = 0.0f;
}

// Constants defining reduction types for clarity
kernel void forward_map_reduce(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant int& input_size         [[buffer(2)]],
    constant uint& reductionType     [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    threadgroup float sharedData[1024];

    // Load data into threadgroup memory
    sharedData[tid] = ((int)tid < input_size) ? input[tid] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction operation
    switch (reductionType) {
        case REDUCTION_SUM:
        case REDUCTION_MEAN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] += sharedData[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = (reductionType == REDUCTION_MEAN) ? (sharedData[0] / input_size) : sharedData[0];
            }
            break;

        case REDUCTION_MAX:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_MIN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_SOFTMAX:
            if (tid == 0) {
                float maxVal = input[0];
                for (int i = 1; i < input_size; ++i) {
                    maxVal = max(maxVal, input[i]);
                }

                float sumExp = 0.0f;
                for (int i = 0; i < input_size; ++i) {
                    sumExp += exp(input[i] - maxVal);
                }

                for (int i = 0; i < input_size; ++i) {
                    output[i] = exp(input[i] - maxVal) / sumExp;
                }
            }
            break;
    }
}

kernel void backward_map_reduce(
    device const float* outputDelta  [[buffer(0)]],
    device const float* forwardOutput[[buffer(1)]],
    device float* inputErrors        [[buffer(2)]],
    constant uint& input_size        [[buffer(3)]],
    constant uint& reductionType     [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= input_size) return;

    switch (reductionType) {
        case REDUCTION_SUM:
            inputErrors[tid] = outputDelta[0];
            break;

        case REDUCTION_MEAN:
            inputErrors[tid] = outputDelta[0] / input_size;
            break;

        case REDUCTION_MAX:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_MIN:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_SOFTMAX:
            // forwardOutput is already softmaxed values here
            float grad = 0.0f;
            for (uint i = 0; i < input_size; ++i) {
                float indicator = (tid == i) ? 1.0f : 0.0f;
                grad += outputDelta[i] * forwardOutput[tid] * (indicator - forwardOutput[i]);
            }
            inputErrors[tid] = grad;
            break;
    }
}

)";

} // namespace multilayerkernels

#pragma endregion Declarations }
#endif
