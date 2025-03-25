#include <metal_stdlib>
using namespace metal;

#include "common.metal"



float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.f;

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
    
    
    // For non-softmax, apply the chosen activation in place
    shared_y[tid] = activate(shared_y[tid], activation);
    
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3) Write the result to the output array
    y[gid] = shared_y[tid];
    
    debug[gid] = y[gid];
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
                                           constant uint&      activation       [[buffer(8)]],
                                           device float*       prevLayerErrors  [[buffer(9)]], // final error to the previous layer's activations
                                           constant uint&      batch_size       [[buffer(10)]],
                                           device atomic_float* gradientsW      [[buffer(11)]],
                                           device atomic_float* gradientsB      [[buffer(12)]],
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
    
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;  // computed gradient (partial dW)
        
        // Write explicitly to gradients buffer instead of modifying W
        uint gradWIdx = i * output_dim + neuron_id;
        atomic_fetch_add_explicit(&gradientsW[gradWIdx], grad, memory_order_relaxed);
        
        // Use existing weights (unchanged) to propagate errors backward
        float weightVal = W[i * output_dim + neuron_id];
        float prevErrTerm = weightVal * delta;
        
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevError[i],
                                  prevErrTerm, memory_order_relaxed);
    }
    
    // Update bias gradient explicitly (no direct bias updates)
    device atomic_float* pGradientsB = (device atomic_float*)&gradientsB[neuron_id];
    atomic_fetch_add_explicit(pGradientsB, delta, memory_order_relaxed);
}


kernel void compute_deltas(
    device const float* y_hat        [[buffer(0)]],  // Predicted outputs
    device const float* y            [[buffer(1)]],  // Targets
    device float*       deltaScratchBuffer [[buffer(2)]], // clearly named scratch buffer
    constant uint&      output_dim   [[buffer(3)]],
    constant uint&      activation   [[buffer(4)]],
    constant uint&      batch_size   [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
)
{
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float pred = y_hat[sample_id * output_dim + neuron_id];
    float target = y[sample_id * output_dim + neuron_id];

    float raw_error = pred - target;
    float delta = activate_derivative(pred, activation) * raw_error;

    // Clamp delta explicitly to ensure numeric stability
    delta = clamp(delta, -threshold, threshold);

    deltaScratchBuffer[sample_id * output_dim + neuron_id] = delta;
}

kernel void accumulate_partial_gradients(
    device const float* h                      [[buffer(0)]],  // Inputs (batch_size × input_dim)
    device const float* deltaScratchBuffer     [[buffer(1)]],  // Deltas (batch_size × output_dim)
    device float*       gradientScratchBuffer  [[buffer(2)]],  // Gradients (input_dim × output_dim), temporary
    constant uint&      input_dim              [[buffer(3)]],
    constant uint&      output_dim             [[buffer(4)]],
    constant uint&      batch_size             [[buffer(5)]],
    uint2               gid                    [[thread_position_in_grid]],
    uint2               tid                    [[thread_position_in_threadgroup]],
    uint2               threads_per_group      [[threads_per_threadgroup]]
)
{
    // Global indices explicitly
    uint input_idx  = gid.x; // input dimension index
    uint output_idx = gid.y; // output dimension index

    if (input_idx >= input_dim || output_idx >= output_dim) return;

    // Partial gradient sum explicitly for this thread
    float partial_gradient = 0.0f;

    // Accumulate gradients explicitly over all samples
    for (uint s = 0; s < batch_size; ++s) {
        float h_val = h[s * input_dim + input_idx];
        float delta_val = deltaScratchBuffer[s * output_dim + output_idx];
        partial_gradient += h_val * delta_val;
    }

    // Index explicitly into gradient scratch buffer
    uint gradient_idx = input_idx * output_dim + output_idx;

    // Write explicitly into global gradient scratch buffer (no atomic needed if one thread per weight)
    gradientScratchBuffer[gradient_idx] = partial_gradient;
}
