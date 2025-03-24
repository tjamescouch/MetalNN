#include <metal_stdlib>
using namespace metal;

#include "common.metal"



float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.f;
constant float epsilon = 1.0e-5f;

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

    // For non-softmax, apply the chosen activation in place
    shared_y[tid] = activate(shared_y[tid], activation);

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


// -----------------------------------------------------------------------------
// Adjust these tile sizes to match your GPU’s best performance characteristics:
#define TILE_W 16
#define TILE_H 16

// Choose a chunk size to process batch samples in local memory at once
// (small enough to fit comfortably in threadgroup memory).
#define CHUNK_SIZE 8u


// -----------------------------------------------------------------------------
// The drop-in replacement kernel.
// NOTE: You must dispatch it 2D, covering (input_dim, output_dim).
kernel void learn_terminal_dense_layer(
    device const float*   h               [[buffer(0)]],  // final layer input activations
    device const float*   W               [[buffer(1)]],  // weights (no direct updates)
    device const float*   b               [[buffer(2)]],  // biases (no direct updates)
    device const float*   y_hat           [[buffer(3)]],  // predicted outputs
    device const float*   y               [[buffer(4)]],  // targets
    device float*         error           [[buffer(5)]],  // partial error (this final layer's delta)
    constant uint&        input_dim       [[buffer(6)]],
    constant uint&        output_dim      [[buffer(7)]],
    constant uint&        activation      [[buffer(8)]],
    device float*         prevLayerErrors [[buffer(9)]],  // error to previous layer
    constant uint&        batch_size      [[buffer(10)]],
    device atomic_float*  gradientsW      [[buffer(11)]], // weights gradients buffer
    device atomic_float*  gradientsB      [[buffer(12)]], // bias gradients buffer
    uint2                 tid             [[thread_position_in_threadgroup]],
    uint2                 gid             [[thread_position_in_grid]]
)
{
    // If each threadgroup covers TILE_W x TILE_H in (i,j) space:
    uint iBlock = gid.x;
    uint jBlock = gid.y;

    uint local_x = tid.x;
    uint local_y = tid.y;

    uint i = iBlock * TILE_W + local_x;
    uint j = jBlock * TILE_H + local_y;

    // Check if (i, j) is within matrix bounds:
    bool valid_i = (i < input_dim);
    bool valid_j = (j < output_dim);

    // -------------------------------------------------------------------------
    // We'll accumulate partial sums for:
    //   gradW(i,j) = sum over s ( h[s,i]* delta[s,j] )
    //   gradB(j)   = sum over s ( delta[s,j] )
    // in registers across the entire batch to avoid repeated atomic adds.

    float accum_dW = 0.0f;
    float accum_dB = 0.0f;

    // -------------------------------------------------------------------------
    // For prevLayerErrors[s,i], we do:
    //   prevLayerErrors[s,i] += sum over j ( delta[s,j] * W[i,j] )
    // but each tile covers a subset of j. We'll add that partial sum in chunks.
    //
    // We'll chunk the batch dimension by CHUNK_SIZE so we can store partial sums
    // for each chunk in local memory. That means far fewer atomic adds overall.

    // A thread will accumulate partial sums for each of up to CHUNK_SIZE samples.
    // Then we do a single atomic add per sample after we've processed that chunk of j.

    // localPrevErr[s_in_chunk][?]:
    // We only need 1 float per sample in the chunk. Each thread accumulates
    // the partial sum for that sample. Then we do an atomic add at the end
    // of the chunk. We'll label it [CHUNK_SIZE], since each thread can handle
    // CHUNK_SIZE samples in that chunk.
    threadgroup float localPrevErr[CHUNK_SIZE];

    // We'll iterate over the batch dimension in increments of CHUNK_SIZE.
    for (uint s_offset = 0; s_offset < batch_size; s_offset += CHUNK_SIZE)
    {
        uint chunkCount = min(CHUNK_SIZE, batch_size - s_offset);

        // ---------------------------------------------------------------------
        // Initialize the local partial sums for each sample in this chunk
        for (uint c = 0; c < chunkCount; c++) {
            localPrevErr[c] = 0.0f;
        }

        // ---------------------------------------------------------------------
        // Barrier to ensure localPrevErr[] is zeroed for all threads
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------------
        // Accumulate partial sums for each sample in the chunk
        for (uint c = 0; c < chunkCount; c++)
        {
            uint s = s_offset + c;

            // 1) Compute delta[s,j] if valid_j (final-layer delta):
            float delta_sj = 0.0f;
            if (valid_j) {
                float outVal   = y_hat[s * output_dim + j];
                float targetVal= y[s * output_dim + j];
                float rawErr   = outVal - targetVal;

                float dAct     = activate_derivative(outVal, activation);
                delta_sj       = clamp(rawErr * dAct, -threshold, threshold);

                // Write the per-sample delta to `error` buffer
                error[s * output_dim + j] = delta_sj;
            }

            // 2) Accumulate into gradW, gradB if valid_i && valid_j
            if (valid_i && valid_j) {
                float h_si  = h[s * input_dim + i];
                accum_dW    += (h_si * delta_sj);
                accum_dB    += delta_sj;
            }

            // 3) Partial sum for prevLayerErrors:
            //    We only handle the contribution from 'this j' in this tile.
            //    localPrevErr[c] will accumulate delta_sj * W[i,j].
            if (valid_i && valid_j) {
                float w_ij   = W[i * output_dim + j];
                localPrevErr[c] += (delta_sj * w_ij);
            }
        }

        // ---------------------------------------------------------------------
        // Wait for all threads in the tile to finish accumulation
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---------------------------------------------------------------------
        // Now we do an atomic add to prevLayerErrors for the partial sums.
        // Each thread has localPrevErr[c], which is the sum from the subset of j
        // that this thread covers (i.e. just one j). We want to combine partial sums
        // across all j in the tile. That implies we need a reduction across local_y
        // within the tile for the same i.
        //
        // A simpler approach (shown here) is: each thread directly does an atomic add
        // for its partial sum. This is 1 atomic add per sample per thread, which is
        // still far fewer than the naive triple loop. If your batch_size and tile
        // dimensions are large, consider a further reduction. But let's keep it direct
        // for clarity so it “just works” and converges.

        if (valid_i) {
            for (uint c = 0; c < chunkCount; c++) {
                float partialVal = localPrevErr[c];
                if (fabs(partialVal) > 0.0f) {
                    uint s = s_offset + c;
                    atomic_fetch_add_explicit(
                        (device atomic_float*) &prevLayerErrors[s * input_dim + i],
                        partialVal,
                        memory_order_relaxed
                    );
                }
            }
        }

        // Barrier in case we need to safely reuse localPrevErr in the next chunk
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // Finally, after processing the entire batch, we do exactly one atomic add
    // for gradW[i,j] and exactly one atomic add for gradB[j] (if valid).
    if (valid_i && valid_j) {
        uint wIdx = i * output_dim + j;
        atomic_fetch_add_explicit(&gradientsW[wIdx], accum_dW, memory_order_relaxed);
    }

    // For gradB[j], let only one "i-lane" in the tile do the bias update.
    // We can pick local_x == 0 for that:
    if (valid_j && (local_x == 0)) {
        atomic_fetch_add_explicit(&gradientsB[j], accum_dB, memory_order_relaxed);
    }
}

