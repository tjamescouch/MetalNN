#include <metal_stdlib>
using namespace metal;

#include "common.metal"



float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 100.0f;
constant float threshold    = 1.f;
constant float GRAD_CLIP_THRESHOLD = 10.0f;

kernel void forward_non_softmax_dense_layer(
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



kernel void backward_non_terminal_non_softmax_dense_layer(
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



kernel void backward_non_terminal_non_softmax_dense_layer_new(
    // Input activations from the previous layer (size = batch_size * input_dim)
    device const float*      h               [[ buffer(0) ]],

    // Weights for this layer (size = input_dim * output_dim)
    device const float*      W               [[ buffer(1) ]],

    // (Optional) Biases for this layer (size = output_dim)
    // Remove if truly unused:
    device const float*      b               [[ buffer(2) ]],

    // Forward outputs (post-activation) from this layer (size = batch_size * output_dim)
    device const float*      y_hat           [[ buffer(3) ]],

    // Incoming error from the next layer (size = batch_size * output_dim)
    device const float*      inputErrors     [[ buffer(4) ]],

    // Output array for this layer’s delta (size = batch_size * output_dim)
    device float*            outputError     [[ buffer(5) ]],

    // Problem dimensions
    constant uint&           input_dim       [[ buffer(6) ]],
    constant uint&           output_dim      [[ buffer(7) ]],
    constant uint&           activation      [[ buffer(8) ]],
    constant uint&           batch_size      [[ buffer(9) ]],

    // Final error back-propagated to the previous layer (size = batch_size * input_dim)
    // Declared as atomic_float* to ensure correct alignment for atomic operations
    device atomic_float*     prevLayerErrors [[ buffer(10) ]],

    // Gradient accumulators for W (size = input_dim * output_dim)
    device atomic_float*     gradientsW      [[ buffer(11) ]],

    // Gradient accumulators for b (size = output_dim)
    device atomic_float*     gradientsB      [[ buffer(12) ]],

    // Thread identifiers
    uint tid                                 [[ thread_position_in_threadgroup ]],
    uint gid                                 [[ thread_position_in_grid       ]]
)
{
    // Compute sample and neuron based on global ID
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    // Bounds check
    if (sample_id >= batch_size || neuron_id >= output_dim) {
        return;
    }

    // Offset pointers to the relevant portion for this sample
    const device float* sample_h         = h      + (sample_id * input_dim);
    const device float* sample_y_hat     = y_hat  + (sample_id * output_dim);
    const device float* sample_inErrors  = inputErrors + (sample_id * output_dim);
          device float* sample_outError  = outputError  + (sample_id * output_dim);

    // Compute raw error and activation derivative
    float raw_error = sample_inErrors[neuron_id];
    float dAct = activate_derivative(sample_y_hat[neuron_id], activation);

    // Delta = error * derivative
    float delta = raw_error * dAct;

    // Gradient clipping to avoid exploding grads
    delta = clamp(delta, -GRAD_CLIP_THRESHOLD, GRAD_CLIP_THRESHOLD);

    // Store this delta in outputError for debugging/inspection
    sample_outError[neuron_id] = delta;

    // Accumulate weight gradients and propagate error backward
    for (uint i = 0; i < input_dim; i++)
    {
        float grad = sample_h[i] * delta;  // partial derivative wrt W[i, neuron_id]
        uint gradWIdx = i * output_dim + neuron_id;

        // Atomically add to the weight gradients
        atomic_fetch_add_explicit(&gradientsW[gradWIdx],
                                  grad,
                                  memory_order_relaxed);

        // Use the current weight to propagate error to prev layer
        float weightVal = W[gradWIdx];
        float prevErrorTerm = weightVal * delta;

        // Atomically add to the error that goes to the previous layer's activations
        // offset = sample_id * input_dim + i
        size_t prevErrorIdx = sample_id * input_dim + i;
        atomic_fetch_add_explicit(&prevLayerErrors[prevErrorIdx],
                                  prevErrorTerm,
                                  memory_order_relaxed);
    }

    // Accumulate bias gradient
    atomic_fetch_add_explicit(&gradientsB[neuron_id],
                              delta,
                              memory_order_relaxed);
}


constant uint CHUNK_SIZE = 128;


kernel void backward_terminal_non_softmax_dense_layer(
    device const float*      inputs            [[buffer(0)]],  // [batch_size * input_dim]
    device const float*      weights           [[buffer(1)]],  // [input_dim * output_dim]
    device const float*      predictions       [[buffer(2)]],  // [batch_size * output_dim]
    device const float*      targets           [[buffer(3)]],  // [batch_size * output_dim]
    device atomic_float*     gradientWeights   [[buffer(4)]],  // [input_dim * output_dim]
    device atomic_float*     gradientBiases    [[buffer(5)]],  // [output_dim]
    device float*            errorsPreviousLayer [[buffer(6)]],// [batch_size * input_dim]
    constant uint&           input_dim         [[buffer(7)]],
    constant uint&           output_dim        [[buffer(8)]],
    constant uint&           batch_size        [[buffer(9)]],
    constant uint&           activation        [[buffer(10)]],

    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
)
{
    // Thread identifies one (input_idx, sample_idx)
    uint input_idx  = gid.x;
    uint sample_idx = gid.y;

    // Bounds check
    if (input_idx >= input_dim || sample_idx >= batch_size) {
        return;
    }

    // Load the input activation for this thread’s (sample, input)
    float inputVal = inputs[sample_idx * input_dim + input_idx];

    // We will accumulate partial gradient sums in small chunks to reduce atomic overhead.
    // gradWLocal[i] -> gradient wrt. W[input_idx, (chunkBase + i)]
    // gradBLocal[i] -> gradient wrt. bias for (chunkBase + i)
    // errorLocal -> running sum of error for the previous layer
    float errorLocal = 0.0f;

    // Loop over output dimension in increments of CHUNK_SIZE
    for (uint chunkStart = 0; chunkStart < output_dim; chunkStart += CHUNK_SIZE)
    {
        // How many outputs remain in this chunk
        uint chunkEnd = min(chunkStart + CHUNK_SIZE, output_dim);
        uint chunkLen = chunkEnd - chunkStart;

        // Private arrays to accumulate gradient sums for this chunk
        float gradWLocal[CHUNK_SIZE];
        float gradBLocal[CHUNK_SIZE];

        // Initialize accumulators
        for (uint i = 0; i < chunkLen; i++) {
            gradWLocal[i] = 0.0f;
            gradBLocal[i] = 0.0f;
        }

        // Accumulate partial gradients
        for (uint out = chunkStart; out < chunkEnd; out++)
        {
            float pred = predictions[sample_idx * output_dim + out];
            float tgt  = targets[sample_idx * output_dim + out];
            float dAct = activate_derivative(pred, activation);

            // Delta for this output
            float delta = (pred - tgt) * dAct;
            // Clip the gradient to avoid exploding updates
            delta = clamp(delta, -GRAD_CLIP_THRESHOLD, GRAD_CLIP_THRESHOLD);

            uint localIdx = out - chunkStart;
            gradWLocal[localIdx] += (inputVal * delta);
            gradBLocal[localIdx] += delta;

            // Also accumulate error to propagate back
            float w = weights[input_idx * output_dim + out];
            errorLocal += w * delta;
        }

        // Write gradient sums to global memory (atomic adds)
        // Minimizes the number of atomics by chunking
        for (uint i = 0; i < chunkLen; i++) {
            uint outIdx = chunkStart + i;

            // Gradient wrt weights
            uint globalWIdx = (input_idx * output_dim) + outIdx;
            atomic_fetch_add_explicit(&gradientWeights[globalWIdx],
                                      gradWLocal[i],
                                      memory_order_relaxed);

            // Gradient wrt bias
            atomic_fetch_add_explicit(&gradientBiases[outIdx],
                                      gradBLocal[i],
                                      memory_order_relaxed);
        }
    }

    // Finally, write the backpropagated error to the previous layer
    errorsPreviousLayer[sample_idx * input_dim + input_idx] = errorLocal;
}
