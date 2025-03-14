#include <metal_stdlib>
using namespace metal;

#include "common.metal"



float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.0f;


kernel void forward_dense_layer(
    device const float* h         [[buffer(0)]],  // Input activations
    device       float* y         [[buffer(1)]],  // Output activations
    device const float* W         [[buffer(2)]],  // Weights
    device const float* b         [[buffer(3)]],  // Biases
    constant uint& hidden_dim     [[buffer(4)]],  // hidden_dim
    constant uint& output_dim     [[buffer(5)]],  // output_dim
    constant uint& activation     [[buffer(6)]],  // Activation type
    constant uint& batchSize      [[buffer(7)]],  // # of samples
    device float* debug           [[buffer(8)]],  // Debug buffer
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
)
{

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
    if (activation == ACTIVATION_SOFTMAX) {
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
        shared_y[tid] = activate(shared_y[tid], activation);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Write the result to the output array
    y[gid] = shared_y[tid];
}



kernel void learn_non_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // input activations
    device float*       W                [[buffer(1)]],  // weights
    device const float* b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // layer outputs
    device const float* inputErrors      [[buffer(4)]],  // errors fed INTO this layer from next
    device float*       outputError      [[buffer(5)]],  // errors fed BACK to previous (delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    constant float&     decay            [[buffer(8)]],  // e.g., momentum or weight decay factor
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // final error to the previous layer's activations
    constant uint&      batch_size       [[buffer(12)]],
    constant float& learning_rate    [[buffer(13)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

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



kernel void learn_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // final layer input activations
    device float*       W                [[buffer(1)]],  // weights
    device float*       b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // predicted outputs
    device const float* y                [[buffer(4)]],  // targets
    device float*       error            [[buffer(5)]],  // partial error (this final layer's delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    constant float&     decay            [[buffer(8)]],
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // error to previous layer
    constant uint&      batch_size       [[buffer(12)]],
    constant float&     learning_rate    [[buffer(13)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

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
