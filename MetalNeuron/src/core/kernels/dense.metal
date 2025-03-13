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
#define ACTIVATION_GELU    5


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.0f;

/**
 * forward_dense_layer
 *
 * This kernel computes the forward pass of a dense (fully-connected) layer.
 *
 * Buffers (in order):
 *  0) h:    The input activations to this layer (shape: [batchSize * hidden_dim]).
 *           Index as h[sample_id * hidden_dim + i].
 *  1) y:    The output activations of this layer (shape: [batchSize * output_dim]).
 *           We'll fill y with activated outputs.
 *  2) W:    The weight matrix (shape: [hidden_dim * output_dim]), row-major:
 *           W[i * output_dim + neuron_id].
 *  3) b:    The bias vector (shape: [output_dim]).
 *  4) pH:   A single uint specifying the hidden_dim (the # of inputs per neuron).
 *  5) pN:   A single uint specifying the output_dim (the # of neurons in this layer).
 *  6) activation: A single uint specifying the activation type (RELU, TANH, etc.).
 *  7) batchSize: The # of samples in the batch (uint).
 *  8) debug: A debug float buffer [batchSize * output_dim], for optional logging if desired.
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 *
 * We typically dispatch with gridSize = batchSize * output_dim,
 * so each thread handles (sample_id, neuron_id).
 */
kernel void forward_dense_layer(
    device const float* h         [[buffer(0)]],  // Input activations
    device       float* y         [[buffer(1)]],  // Output activations
    device const float* W         [[buffer(2)]],  // Weights
    device const float* b         [[buffer(3)]],  // Biases
    device const uint* pH         [[buffer(4)]],  // hidden_dim
    device const uint* pN         [[buffer(5)]],  // output_dim
    device const uint* activation [[buffer(6)]],  // Activation type
    constant uint& batchSize      [[buffer(7)]],  // # of samples
    device float* debug           [[buffer(8)]],  // Debug buffer
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
)
{
    uint hidden_dim = *pH;
    uint output_dim = *pN;
    uint act_type   = *activation;

    // sample_id, neuron_id
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batchSize || neuron_id >= output_dim) return;

    // We'll use a threadgroup array for partial storage
    threadgroup float shared_y[1024];

    // 1) Compute the pre-activation sum
    float sum = b[neuron_id];
    for (uint i = 0; i < hidden_dim; ++i) {
        float inputVal = h[sample_id * hidden_dim + i];
        float weightVal = W[i * output_dim + neuron_id];
        sum += inputVal * weightVal;
    }

    // Clamp to avoid numerical blow-up
    sum = clamp(sum, -max_abs_sum, max_abs_sum);

    // Store in threadgroup memory to do optional softmax
    shared_y[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) If activation is softmax, handle it per sample
    if (act_type == ACTIVATION_SOFTMAX) {
        // We do a per-sample softmax in threadgroup memory
        uint sample_offset = sample_id * output_dim;
        // Find max for numeric stability
        float maxVal = shared_y[sample_offset];  // We assume sample_offset+0 is in range
        for (uint i = 1; i < output_dim; ++i) {
            float val = shared_y[sample_offset + i];
            maxVal = max(maxVal, val);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exponentiate each
        shared_y[tid] = exp(shared_y[tid] - maxVal);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One thread (neuron_id=0) sums and normalizes
        if (neuron_id == 0) {
            float sumExp = 0.0f;
            for (uint i = 0; i < output_dim; ++i) {
                sumExp += shared_y[sample_offset + i];
            }
            // Normalize
            for (uint i = 0; i < output_dim; ++i) {
                shared_y[sample_offset + i] /= sumExp;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Optionally store debug
        debug[gid] = shared_y[tid];
    }
    else {
        // For non-softmax, apply the chosen activation in place
        shared_y[tid] = activate(shared_y[tid], act_type);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Write the result to the output array
    y[gid] = shared_y[tid];
}


/**
 * learn_non_terminal_dense_layer
 *
 * This kernel handles the backward pass for a dense layer that is NOT the final
 * layer (i.e., it receives "input errors" from the next layer).
 *
 * Buffers (in order):
 *  0) h:             The input activations fed into this dense layer [batchSize * input_dim].
 *  1) W:             The weight matrix for this layer [input_dim * output_dim].
 *  2) b:             The bias vector [output_dim].
 *  3) y_hat:         The output activations from this layer [batchSize * output_dim].
 *  4) inputErrors:   The "errors" fed INTO the current layer from the next layer [batchSize * output_dim].
 *                    (Sometimes called dY or "output gradient of next layer.")
 *  5) outputError:   The "errors" fed BACK by this layer to the previous layer [batchSize * output_dim].
 *                    However, here we only store each neuron's partial ∂L/∂(neuron output).
 *  6) input_dim:     # of inputs per neuron (uint).
 *  7) output_dim:    # of neurons in this layer (uint).
 *  8) pDecay:        A float pointer for weight decay or momentum factor (decay).
 *  9) activation:    The activation type for this layer (uint).
 * 10) debug:         Debug buffer [batchSize * output_dim].
 * 11) prevLayerErrors: The "output errors" from this layer to the PREVIOUS layer [batchSize * input_dim].
 *                      (i.e., ∂L/∂(h), shape [batchSize * input_dim]).
 * 12) batch_size:    The number of samples in the batch (uint).
 * 13) pLearningRate: A float pointer giving the learning rate.
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 *
 * Typically, dispatch with gridSize = batchSize * output_dim.
 * Each thread corresponds to (sample_id, neuron_id).
 */
kernel void learn_non_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // input activations
    device float*       W                [[buffer(1)]],  // weights
    device const float* b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // layer outputs
    device const float* inputErrors      [[buffer(4)]],  // errors fed INTO this layer from next
    device float*       outputError      [[buffer(5)]],  // errors fed BACK to previous (delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    device float*       pDecay           [[buffer(8)]],  // e.g., momentum or weight decay factor
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // final error to the previous layer's activations
    constant uint&      batch_size       [[buffer(12)]],
    device const float* pLearningRate    [[buffer(13)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay         = *pDecay;
    float learning_rate = *pLearningRate;

    // Pointers to this sample's portion of each buffer
    const device float* sample_h         = h + (sample_id * input_dim);
    const device float* sample_y_hat     = y_hat + (sample_id * output_dim);
    const device float* sample_inErrors  = inputErrors + (sample_id * output_dim);
    device float*       sample_outError  = outputError + (sample_id * output_dim);
    device float*       sample_prevError = prevLayerErrors + (sample_id * input_dim);

    // raw_error is the incoming gradient wrt this neuron's output
    float raw_error = sample_inErrors[neuron_id];

    // Delta = error * activation derivative
    float dAct = activate_derivative(sample_y_hat[neuron_id], activation);
    float delta = raw_error * dAct;

    // Clamp to avoid numerical blow-up
    delta = clamp(delta, -threshold, threshold);

    // Save partial delta in outputError (i.e., error from this layer's viewpoint)
    sample_outError[neuron_id] = delta;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For each input in [0..input_dim-1], update weight and accumulate error
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;  // partial dW

        // clamp grad
        //grad = clamp(grad, -threshold, threshold);


        debug[i * output_dim + neuron_id] = grad;

        // We want to read the old weight, THEN do an atomic add
        device atomic_float* wAddr = (device atomic_float*)&W[i * output_dim + neuron_id];

        float oldWeight = atomic_load_explicit(wAddr, memory_order_relaxed);

        // Atomic update the weight
        // W -= learning_rate * grad * decay
        float wUpdate = -learning_rate * grad * decay;
        atomic_fetch_add_explicit(wAddr, wUpdate, memory_order_relaxed);

        // For the error to pass back to previous layer: oldWeight * delta
        // Use oldWeight from before we updated it.
        float prevErrTerm = oldWeight * delta;
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevError[i],
                                  prevErrTerm, memory_order_relaxed);
    }

    // Update bias
    device atomic_float* bAddr = (device atomic_float*)&b[neuron_id];
    float biasUpdate = -learning_rate * delta * decay;
    atomic_fetch_add_explicit(bAddr, biasUpdate, memory_order_relaxed);
}


/**
 * learn_terminal_dense_layer
 *
 * This kernel handles the backward pass for a dense layer that IS the final layer.
 * Instead of "inputErrors," it receives the direct target difference (y_hat - y)
 * or some derivative thereof. We call that "inputErrors" in a final layer, but
 * you have it as the "targets" array. We'll keep to your naming scheme:
 *
 * Buffers (in order):
 *  0) h:      input activations for this final layer [batchSize * input_dim].
 *  1) W:      the weight matrix [input_dim * output_dim].
 *  2) b:      the bias vector [output_dim].
 *  3) y_hat:  the predicted outputs for this layer [batchSize * output_dim].
 *  4) y:      the ground-truth targets [batchSize * output_dim].
 *  5) error:  the partial errors from this final layer's perspective [batchSize * output_dim].
 *  6) input_dim:  # of inputs per neuron (uint).
 *  7) output_dim: # of neurons in this layer (uint).
 *  8) pDecay:     float pointer for weight decay or momentum factor.
 *  9) activation: the activation type for this final layer (uint).
 * 10) debug:      debug buffer [batchSize * output_dim].
 * 11) prevLayerErrors: the "output errors" from this final layer to the previous layer [batchSize * input_dim].
 * 12) batch_size: how many samples in the batch (uint).
 * 13) pLearningRate: the learning rate (float pointer).
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 */
kernel void learn_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // final layer input activations
    device float*       W                [[buffer(1)]],  // weights
    device float*       b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // predicted outputs
    device const float* y                [[buffer(4)]],  // targets
    device float*       error            [[buffer(5)]],  // partial error (this final layer's delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    device float*       pDecay           [[buffer(8)]],
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // error to previous layer
    constant uint&      batch_size       [[buffer(12)]],
    device const float* pLearningRate    [[buffer(13)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay         = *pDecay;
    float learning_rate = *pLearningRate;

    // Pointers to this sample's data
    const device float* sample_h     = h     + (sample_id * input_dim);
    const device float* sample_y_hat = y_hat + (sample_id * output_dim);
    const device float* sample_y     = y     + (sample_id * output_dim);
    device float*       sample_error = error + (sample_id * output_dim);
    device float*       sample_prevE = prevLayerErrors + (sample_id * input_dim);

    // For terminal layer, raw_error is typically (y_hat - y).
    // We'll clamp, then multiply by derivative if not softmax.
    float raw_error = sample_y_hat[neuron_id] - sample_y[neuron_id];
    debug[gid] = raw_error;

    float delta;
    if (activation == ACTIVATION_SOFTMAX) {
        // In typical frameworks, for cross-entropy + softmax => delta = (y_hat - y).
        delta = raw_error;
    } else {
        // Multiply by derivative of the activation
        float dAct = activate_derivative(sample_y_hat[neuron_id], activation);
        delta = raw_error * dAct;
    }

    // clamp
    delta = clamp(delta, -threshold, threshold);
    // Save in error buffer
    sample_error[neuron_id] = delta;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weight + bias update
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;
        grad = clamp(grad, -threshold, threshold);

        device atomic_float* wAddr = (device atomic_float*)&W[i * output_dim + neuron_id];
        float oldWeight = atomic_load_explicit(wAddr, memory_order_relaxed);

        float wUpdate = -learning_rate * grad * decay;
        atomic_fetch_add_explicit(wAddr, wUpdate, memory_order_relaxed);

        // error to feed back
        float prevErrTerm = oldWeight * delta;
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevE[i], prevErrTerm, memory_order_relaxed);
    }

    // Bias
    device atomic_float* bAddr = (device atomic_float*)&b[neuron_id];
    float biasUpdate = -learning_rate * delta * decay;
    atomic_fetch_add_explicit(bAddr, biasUpdate, memory_order_relaxed);
}
