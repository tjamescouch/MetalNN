#include <metal_stdlib>

#include "common.metal"

using namespace metal;


kernel void forward_self_attention(
    device const float* input          [[buffer(0)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ       [[buffer(1)]],  // [inputDim, modelDim]
    device const float* weightsK       [[buffer(2)]],  // [inputDim, modelDim]
    device const float* weightsV       [[buffer(3)]],  // [inputDim, modelDim]
    device const float* weightsO       [[buffer(4)]],  // [modelDim, inputDim]
    device float* output               [[buffer(5)]],  // [batchSize, seqLength, inputDim]
    constant uint& batchSize           [[buffer(6)]],
    constant uint& seqLength           [[buffer(7)]],
    constant uint& inputDim            [[buffer(8)]],
    constant uint& modelDim            [[buffer(9)]],
    uint gid                           [[thread_position_in_grid]])
{
    uint totalElements = batchSize * seqLength * modelDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * modelDim);
    uint seqIdx = (gid / modelDim) % seqLength;
    uint dimIdx = gid % modelDim;

    // Compute linear projections for Q, K, V at current position
    float Q = 0.0f, K = 0.0f, V = 0.0f;

    for (uint i = 0; i < inputDim; ++i) {
        float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
        Q += inputVal * weightsQ[i * modelDim + dimIdx];
        K += inputVal * weightsK[i * modelDim + dimIdx];
        V += inputVal * weightsV[i * modelDim + dimIdx];
    }

    // Compute attention scores across all positions in sequence
    float scale = sqrt((float)modelDim);
    float attentionSum = 0.0f;
    float normalizationFactor = 0.0f;

    for (uint j = 0; j < seqLength; ++j) {
        float Kj = 0.0f;
        // Compute K projection for position j
        for (uint d = 0; d < inputDim; ++d) {
            float inputValK = input[batchIdx * seqLength * inputDim + j * inputDim + d];
            Kj += inputValK * weightsK[d * modelDim + dimIdx];
        }
        float attentionScore = (Q * Kj) / scale;
        float attentionWeight = exp(attentionScore); // Unnormalized softmax component
        normalizationFactor += attentionWeight;

        // Compute V projection for position j
        float Vj = 0.0f;
        for (uint d = 0; d < inputDim; ++d) {
            float inputValV = input[batchIdx * seqLength * inputDim + j * inputDim + d];
            Vj += inputValV * weightsV[d * modelDim + dimIdx];
        }

        attentionSum += attentionWeight * Vj;
    }

    attentionSum /= normalizationFactor;  // Softmax normalization step explicitly

    // Final output projection back to input dimension
    for (uint outDim = 0; outDim < inputDim; ++outDim) {
        float projectedOutput = attentionSum * weightsO[dimIdx * inputDim + outDim];
        output[batchIdx * seqLength * inputDim + seqIdx * inputDim + outDim] = projectedOutput;
    }
}

kernel void backward_self_attention(
    device const float* outputErrors    [[buffer(0)]], // [batchSize, seqLength, inputDim]
    device const float* input           [[buffer(1)]], // [batchSize, seqLength, inputDim]
    device const float* weightsQ        [[buffer(2)]], // [inputDim, modelDim]
    device const float* weightsK        [[buffer(3)]], // [inputDim, modelDim]
    device const float* weightsV        [[buffer(4)]], // [inputDim, modelDim]
    device const float* weightsO        [[buffer(5)]], // [modelDim, inputDim]
    device atomic_float* inputErrors    [[buffer(6)]], // [batchSize, seqLength, inputDim]
    device atomic_float* gradWeightsQ   [[buffer(7)]], // [inputDim, modelDim]
    device atomic_float* gradWeightsK   [[buffer(8)]], // [inputDim, modelDim]
    device atomic_float* gradWeightsV   [[buffer(9)]], // [inputDim, modelDim]
    device atomic_float* gradWeightsO   [[buffer(10)]],// [modelDim, inputDim]
    constant uint& batchSize            [[buffer(11)]],
    constant uint& seqLength            [[buffer(12)]],
    constant uint& inputDim             [[buffer(13)]],
    constant uint& modelDim             [[buffer(14)]],
    uint gid                            [[thread_position_in_grid]])
{
    uint totalElements = batchSize * seqLength * inputDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * inputDim);
    uint seqIdx   = (gid / inputDim) % seqLength;
    uint dimIdx   = gid % inputDim;

    float error = outputErrors[gid];

    // Compute gradients w.r.t. output projection weights (weightsO)
    for (uint mDim = 0; mDim < modelDim; ++mDim) {
        atomic_fetch_add_explicit(
            &gradWeightsO[mDim * inputDim + dimIdx],
            error,
            memory_order_relaxed
        );
    }

    float dAttentionSum = 0.0f;

    // Propagate error through output projection weights to attention sum
    for (uint mDim = 0; mDim < modelDim; ++mDim) {
        dAttentionSum += error * weightsO[mDim * inputDim + dimIdx];
    }

    float scale = sqrt((float)modelDim);

    // Iterate over all positions to compute attention gradients
    for (uint j = 0; j < seqLength; ++j) {
        float Qj = 0.0f, Kj = 0.0f, Vj = 0.0f;

        // Compute Q, K, V at position j
        for (uint d = 0; d < inputDim; ++d) {
            float inputVal = input[batchIdx * seqLength * inputDim + j * inputDim + d];
            Qj += inputVal * weightsQ[d * modelDim + dimIdx];
            Kj += inputVal * weightsK[d * modelDim + dimIdx];
            Vj += inputVal * weightsV[d * modelDim + dimIdx];
        }

        // Compute attention weights and derivatives
        float attentionScore = (Qj * Kj) / scale;
        float attentionWeight = exp(attentionScore);

        float dAttentionWeight = Vj * dAttentionSum;
        float dAttentionScore = attentionWeight * dAttentionWeight / scale;

        // Gradients for weightsQ, weightsK, weightsV and input errors
        for (uint d = 0; d < inputDim; ++d) {
            float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + d];

            float gradQ = inputVal * dAttentionScore * Kj;
            float gradK = inputVal * dAttentionScore * Qj;
            float gradV = inputVal * attentionWeight * dAttentionSum;

            atomic_fetch_add_explicit(
                &gradWeightsQ[d * modelDim + dimIdx],
                gradQ,
                memory_order_relaxed
            );

            atomic_fetch_add_explicit(
                &gradWeightsK[d * modelDim + dimIdx],
                gradK,
                memory_order_relaxed
            );

            atomic_fetch_add_explicit(
                &gradWeightsV[d * modelDim + dimIdx],
                gradV,
                memory_order_relaxed
            );

            // Compute input errors and propagate
            float dInput = weightsQ[d * modelDim + dimIdx] * dAttentionScore * Kj +
                           weightsK[d * modelDim + dimIdx] * dAttentionScore * Qj +
                           weightsV[d * modelDim + dimIdx] * attentionWeight * dAttentionSum;

            atomic_fetch_add_explicit(
                &inputErrors[batchIdx * seqLength * inputDim + seqIdx * inputDim + d],
                dInput,
                memory_order_relaxed
            );
        }
    }
}
