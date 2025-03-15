#include <metal_stdlib>

#include "common.metal"

using namespace metal;

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
    uint totalElements = batchSize * seqLength * inputDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * inputDim);
    uint seqIdx   = (gid / inputDim) % seqLength;
    uint inputIdx = gid % inputDim;

    float scale = sqrt(float(modelDim));

    // Step 1: Compute and store Q, K, V vectors for all positions (if not already computed)
    for (uint d = 0; d < modelDim; ++d) {
        float Q = 0.0f, K = 0.0f, V = 0.0f;
        for (uint i = 0; i < inputDim; ++i) {
            float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
            Q += inputVal * weightsQ[i * modelDim + d];
            K += inputVal * weightsK[i * modelDim + d];
            V += inputVal * weightsV[i * modelDim + d];
        }
        bufferQ[batchIdx * seqLength * modelDim + seqIdx * modelDim + d] = Q;
        bufferK[batchIdx * seqLength * modelDim + seqIdx * modelDim + d] = K;
        bufferV[batchIdx * seqLength * modelDim + seqIdx * modelDim + d] = V;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute attention scores (proper dot product across modelDim)
    float attentionSum = 0.0f;
    float normalizationFactor = 0.0f;

    for (uint j = 0; j < seqLength; ++j) {
        // Compute dot product Q(seqIdx) • K(j)
        float score = 0.0f;
        for (uint d = 0; d < modelDim; ++d) {
            float q_val = bufferQ[batchIdx * seqLength * modelDim + seqIdx * modelDim + d];
            float k_val = bufferK[batchIdx * seqLength * modelDim + j * modelDim + d];
            score += q_val * k_val;
        }

        score /= scale;
        float attentionWeight = exp(score);
        normalizationFactor += attentionWeight;

        // Weighted sum of V(j) for current input dimension (inputIdx)
        float v_sum = 0.0f;
        for (uint d = 0; d < modelDim; ++d) {
            float v_val = bufferV[batchIdx * seqLength * modelDim + j * modelDim + d];
            float weightO = weightsO[d * inputDim + inputIdx];
            v_sum += v_val * weightO;
        }
        attentionSum += attentionWeight * v_sum;
    }

    // Step 3: Normalize (softmax)
    attentionSum /= normalizationFactor;

    // Step 4: Write explicitly to output buffer
    output[batchIdx * seqLength * inputDim + seqIdx * inputDim + inputIdx] = attentionSum;
}

kernel void backward_self_attention(
    device const float* input                        [[buffer(0)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ                     [[buffer(1)]],  // [inputDim, modelDim]
    device const float* weightsK                     [[buffer(2)]],
    device const float* weightsV                     [[buffer(3)]],
    device const float* weightsO                     [[buffer(4)]],

    device const float* bufferQ                      [[buffer(5)]],  // intermediate Q from forward pass
    device const float* bufferK                      [[buffer(6)]],  // intermediate K buffer
    device const float* bufferV                      [[buffer(7)]],  // intermediate V buffer from forward

    device atomic_float* outputErrors                [[buffer(8)]],  // errors leaving this layer (to previous)
    device const float* inputErrors                  [[buffer(9)]],  // errors from next layer

    device atomic_float* gradWeightsQ                [[buffer(10)]],
    device atomic_float* gradWeightsK                [[buffer(11)]],
    device atomic_float* gradWeightsV                [[buffer(12)]],
    device atomic_float* gradWeightsO                [[buffer(13)]],

    constant uint& batchSize                         [[buffer(14)]],
    constant uint& seqLength                         [[buffer(15)]],
    constant uint& inputDim                          [[buffer(16)]],
    constant uint& modelDim                          [[buffer(17)]],

    uint gid                                  [[thread_position_in_grid]]
)
{
    uint totalElements = batchSize * seqLength * modelDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * modelDim);
    uint seqIdx   = (gid / modelDim) % seqLength;
    uint dimIdx   = gid % modelDim;

    float scale = sqrt((float)modelDim);

    // Load the gradient flowing into this layer from the next layer
    float dOutput = inputErrors[batchIdx * seqLength * modelDim + seqIdx * modelDim + dimIdx];

    // Accumulate gradient for output projection weights (O) and propagate errors backward
    for (uint i = 0; i < inputDim; ++i) {
        float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
        float gradO = dOutput * inputVal;

        atomic_fetch_add_explicit(&gradWeightsO[dimIdx * inputDim + i], gradO, memory_order_relaxed);

        // propagate error backward to the previous layer explicitly
        float propagatedError = dOutput * weightsO[dimIdx * inputDim + i];
        atomic_fetch_add_explicit(
            &outputErrors[batchIdx * seqLength * inputDim + seqIdx * inputDim + i],
            propagatedError,
            memory_order_relaxed
        );
    }

    // Gradients for Q, K, V weights (simplified gradient accumulation)
    for (uint i = 0; i < inputDim; ++i) {
        float gradQ = 0.0f;
        float gradK = 0.0f;
        float gradV = 0.0f;

        float q_current_dim = bufferQ[batchIdx * seqLength * modelDim + seqIdx * modelDim + dimIdx];

        for (uint j = 0; j < seqLength; ++j) {
            float score = 0.0f;

            // Compute attention scores (Q • K)
            for (uint d = 0; d < modelDim; ++d) {
                float q_element = bufferQ[batchIdx * seqLength * modelDim + seqIdx * modelDim + d];
                float k_element = bufferK[batchIdx * seqLength * modelDim + j * modelDim + d];
                score += q_element * k_element;
            }

            score /= scale;
            float attentionWeight = exp(score);  // Simplified (softmax normalization pending)

            float k_current_dim = bufferK[batchIdx * seqLength * modelDim + j * modelDim + dimIdx];
            float v_current_dim = bufferV[batchIdx * seqLength * modelDim + j * modelDim + dimIdx];

            float dAttention = attentionWeight * dOutput;

            // Accumulate gradients explicitly using properly defined variables
            gradQ += dAttention * k_current_dim / scale;
            gradK += dAttention * q_current_dim / scale;
            gradV += dAttention * v_current_dim;
        }

        // Explicitly accumulate gradients atomically after loop
        atomic_fetch_add_explicit(&gradWeightsQ[dimIdx], gradQ, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradWeightsK[dimIdx], gradK, memory_order_relaxed);
        atomic_fetch_add_explicit(&gradWeightsV[dimIdx], gradV, memory_order_relaxed);
    }
}
