#include <metal_stdlib>
using namespace metal;

#include "common.metal"



float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.f;
constant float epsilon = 1.0e-5f;

kernel void forward_dense_softmax_layer(
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
        float denominator = abs(sumExp) > epsilon ? sumExp : epsilon;
        for (uint i = 0; i < output_dim; ++i) {
            shared_y[sample_offset + i] /= denominator;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3) Write the result to the output array
    y[gid] = shared_y[tid];
    
    debug[gid] = y[gid];
}



kernel void learn_non_terminal_dense_softmax_layer(
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


kernel void learn_terminal_dense_softmax_layer(
    device const float*  h               [[buffer(0)]],  // final layer input activations
    device const float*  W               [[buffer(1)]],  // weights (no direct updates)
    device const float*  b               [[buffer(2)]],  // biases (no direct updates)
    device const float*  y_hat           [[buffer(3)]],  // predicted outputs
    device const float*  y               [[buffer(4)]],  // targets
    device float*        error           [[buffer(5)]],  // partial error (this final layer's delta)
    constant uint&       input_dim       [[buffer(6)]],
    constant uint&       output_dim      [[buffer(7)]],
    constant uint&       activation      [[buffer(8)]],
    device float*        prevLayerErrors [[buffer(9)]],  // error to previous layer
    constant uint&       batch_size      [[buffer(10)]],
    device atomic_float* gradientsW      [[buffer(11)]], // weights gradients buffer
    device atomic_float* gradientsB      [[buffer(12)]], // bias gradients buffer
    uint                 tid             [[thread_position_in_threadgroup]],
    uint                 gid             [[thread_position_in_grid]]
)
{
    // Identify the current sample and neuron from the global ID
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;
    
    // If out of bounds, exit immediately
    if (sample_id >= batch_size || neuron_id >= output_dim) {
        return;
    }

    // Pointers into per-sample data
    const device float* sample_h     = h     + sample_id * input_dim;
    const device float* sample_y_hat = y_hat + sample_id * output_dim;
    const device float* sample_y     = y     + sample_id * output_dim;
    
    // Write locations in global memory
    device float* sample_error = error + (sample_id * output_dim);
    device float* sample_prevE = prevLayerErrors + (sample_id * input_dim);
    
    // Compute output layer error: delta = (y_hat - y)
    float raw_error = sample_y_hat[neuron_id] - sample_y[neuron_id];
    
    // Clamp if necessary
    float delta = clamp(raw_error, -threshold, threshold);
    
    // Store delta in error buffer (unique address per thread)
    sample_error[neuron_id] = delta;

    // --- No threadgroup_barrier(...) needed here. ---

    // Update weights gradients and accumulate errors for previous layer
    for (uint i = 0; i < input_dim; i++)
    {
        // gradient for W[i, neuron_id] = h[i] * delta
        //float grad = clamp(sample_h[i] * delta, -threshold, threshold);
        float grad = sample_h[i] * delta;
        uint  gradWIdx = i * output_dim + neuron_id;
        
        atomic_fetch_add_explicit(&gradientsW[gradWIdx], grad, memory_order_relaxed);

        // Propagate error to previous layer
        float weightVal     = W[gradWIdx];
        float prevErrTerm   = weightVal * delta;
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevE[i], prevErrTerm, memory_order_relaxed);
    }
    
    // Update bias gradient
    atomic_fetch_add_explicit(&gradientsB[neuron_id], delta, memory_order_relaxed);
}
