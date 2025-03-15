#include <metal_stdlib>

#include "common.metal"

using namespace metal;

// Choose maximum dimensions so you don't overflow thread stack memory.
// You must ensure at runtime: seqLength <= MAX_SEQ_LENGTH && modelDim <= MAX_MODEL_DIM.
#define MAX_SEQ_LENGTH 1024
#define MAX_MODEL_DIM 1024

kernel void forward_self_attention(
    device const float* input                [[buffer(0)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ             [[buffer(1)]],  // [inputDim, modelDim]
    device const float* weightsK             [[buffer(2)]],  // [inputDim, modelDim]
    device const float* weightsV             [[buffer(3)]],  // [inputDim, modelDim]
    device const float* weightsO             [[buffer(4)]],  // [modelDim, inputDim]

    device float* bufferQ                    [[buffer(5)]],  // [batchSize, seqLength, modelDim]
    device float* bufferK                    [[buffer(6)]],  // [batchSize, seqLength, modelDim]
    device float* bufferV                    [[buffer(7)]],  // [batchSize, seqLength, modelDim]


    device float* output                     [[buffer(8)]],  // [batchSize, seqLength, inputDim]

    constant uint& batchSize                 [[buffer(9)]],
    constant uint& seqLength                 [[buffer(10)]],
    constant uint& inputDim                  [[buffer(11)]],
    constant uint& modelDim                  [[buffer(12)]],

    uint gid                                 [[thread_position_in_grid]])
{
    if (gid >= batchSize * seqLength) return;

    uint batchIdx = gid / seqLength;
    uint seqIdx = gid % seqLength;

    // Offset calculations
    uint input_offset = (batchSize * seqIdx + batchIdx) * inputDim;
    uint buffer_offset = (batchSize * seqIdx + batchIdx) * modelDim;
    uint output_offset = input_offset;

    // Step 1: Compute Q, K, V vectors
    for (uint m = 0; m < modelDim; ++m) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float v_sum = 0.0f;

        for (uint i = 0; i < inputDim; ++i) {
            float in_val = input[input_offset + i];
            q_sum += in_val * weightsQ[i * modelDim + m];
            k_sum += in_val * weightsK[i * modelDim + m];
            v_sum += in_val * weightsV[i * modelDim + m];
        }
        bufferQ[buffer_offset + m] = q_sum;
        bufferK[buffer_offset + m] = k_sum;
        bufferV[buffer_offset + m] = v_sum;
    }

    threadgroup float attention_scores[512]; // Adjust size if necessary

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute Attention Scores (Scaled Dot-Product)
    float scale = rsqrt(float(modelDim));

    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        float attn_score = 0.0f;
        for (uint m = 0; m < modelDim; ++m) {
            float q_val = bufferQ[(batchIdx * seqLength + seqIdx) * modelDim + m];
            float k_val = bufferK[(batchIdx * seqLength + seqIdx2) * modelDim + m];
            attn_score += q_val * k_val;
        }
        attn_score *= scale;
        attention_scores[seqIdx2] = attn_score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Softmax on attention scores
    float max_score = attention_scores[0];
    for (uint i = 1; i < seqLength; ++i)
        max_score = max(max_score, attention_scores[i]);

    float sum_exp = 0.0f;
    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        attention_scores[seqIdx2] = exp(attention_scores[seqIdx2] - max_score);
        sum_exp += attention_scores[seqIdx2];
    }

    for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
        attention_scores[seqIdx2] /= sum_exp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Weighted sum of V vectors
    float context[MAX_MODEL_DIM]; // allocate context array on stack
    for (uint m = 0; m < modelDim; ++m) {
        float context_sum = 0.0f;
        for (uint seqIdx2 = 0; seqIdx2 < seqLength; ++seqIdx2) {
            float attn = attention_scores[seqIdx2];
            float v_val = bufferV[(batchIdx * seqLength + seqIdx2) * modelDim + m];
            context_sum += attn * v_val;
        }
        context[m] = context_sum;
    }

    // Step 4b: Write context back to original space using weightsO
    for (uint i = 0; i < inputDim; ++i) {
        float out_sum = 0.0f;
        for (uint m = 0; m < modelDim; ++m) {
            out_sum += context[m] * weightsO[m * inputDim + i];
        }
        output[output_offset + i] = out_sum;
    }
}



kernel void backward_self_attention(
    // Forward inputs
    device const float* input               [[buffer(0)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ            [[buffer(1)]],  // [inputDim, modelDim]
    device const float* weightsK            [[buffer(2)]],  // [inputDim, modelDim]
    device const float* weightsV            [[buffer(3)]],  // [inputDim, modelDim]
    device const float* weightsO            [[buffer(4)]],  // [modelDim, inputDim]

    // Saved forward pass results
    device const float* bufferQ             [[buffer(5)]],  // [batchSize, seqLength, modelDim]
    device const float* bufferK             [[buffer(6)]],  // [batchSize, seqLength, modelDim]
    device const float* bufferV             [[buffer(7)]],  // [batchSize, seqLength, modelDim]

    // Gradients we want to fill/accumulate
    device atomic_float* inputErrors        [[buffer(8)]],  // same shape as input
    device const float*  outputErrors       [[buffer(9)]],  // [batchSize, seqLength, inputDim]

    device atomic_float* gradWeightsQ       [[buffer(10)]], // same shape as weightsQ
    device atomic_float* gradWeightsK       [[buffer(11)]],
    device atomic_float* gradWeightsV       [[buffer(12)]],
    device atomic_float* gradWeightsO       [[buffer(13)]],

    constant uint& batchSize                [[buffer(14)]],
    constant uint& seqLength                [[buffer(15)]],
    constant uint& inputDim                 [[buffer(16)]],
    constant uint& modelDim                 [[buffer(17)]],

    uint gid                                [[thread_position_in_grid]])
{
    //TODO
}
