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
    uint totalElements = batchSize * seqLength * modelDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * modelDim);
    uint seqIdx   = (gid / modelDim) % seqLength;
    uint dimIdx   = gid % modelDim;

    // Step 1: Compute Q, K, V projections explicitly and store them in intermediate buffers
    float Q = 0.0f, K = 0.0f, V = 0.0f;
    for (uint i = 0; i < inputDim; ++i) {
        float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
        Q += inputVal * weightsQ[i * modelDim + dimIdx];
        K += inputVal * weightsK[i * modelDim + dimIdx];
        V += inputVal * weightsV[i * modelDim + dimIdx];
    }

    bufferQ[gid] = Q;
    bufferK[batchIdx * seqLength * modelDim + seqIdx * modelDim + dimIdx] = K;
    bufferV[batchIdx * seqLength * modelDim + seqIdx * modelDim + dimIdx] = V;

    threadgroup float normalizationFactor;
    normalizationFactor = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = sqrt(float(modelDim));
    float attentionSum = 0.0f;

    // Compute attention across the sequence
    for (uint j = 0; j < seqLength; ++j) {
        float Kj = bufferK[batchIdx * seqLength * modelDim + j * modelDim + dimIdx];
        float Vj = bufferV[batchIdx * seqLength * modelDim + j * modelDim + dimIdx];

        float score = (Q * Kj) / scale;
        float attentionWeight = exp(score);

        normalizationFactor += attentionWeight;
        attentionSum += attentionWeight * Vj;
    }

    // Softmax normalization explicitly applied
    attentionSum /= normalizationFactor;

    // Final projection back to inputDim using weightsO
    float projectedOutput = 0.0f;
    for (uint i = 0; i < inputDim; ++i) {
        projectedOutput = attentionSum * weightsO[dimIdx * inputDim + i];
        output[batchIdx * seqLength * inputDim + seqIdx * inputDim + i] = projectedOutput;
    }
}

kernel void backward_self_attention(
    device const float* outputErrors   [[buffer(0)]],  // [batchSize, seqLength, modelDim]
    device const float* input          [[buffer(1)]],  // [batchSize, seqLength, inputDim]
    device const float* weightsQ       [[buffer(2)]],  // [inputDim, modelDim]
    device const float* weightsK       [[buffer(3)]],  // [inputDim, modelDim]
    device const float* weightsV       [[buffer(4)]],  // [inputDim, modelDim]
    device const float* weightsO       [[buffer(5)]],  // [modelDim, inputDim]

    device atomic_float* inputErrors   [[buffer(6)]],  // [batchSize, seqLength, inputDim]
    device atomic_float* gradWeightsQ  [[buffer(7)]],  // [inputDim, modelDim]
    device atomic_float* gradWeightsK  [[buffer(8)]],  // [inputDim, modelDim]
    device atomic_float* gradWeightsV  [[buffer(9)]],  // [inputDim, modelDim]
    device atomic_float* gradWeightsO  [[buffer(10)]], // [modelDim, inputDim]

    constant uint& batchSize           [[buffer(11)]],
    constant uint& seqLength           [[buffer(12)]],
    constant uint& inputDim            [[buffer(13)]],
    constant uint& modelDim            [[buffer(14)]],

    uint gid                           [[thread_position_in_grid]])
{
    uint totalElements = batchSize * seqLength * modelDim;
    if (gid >= totalElements) return;

    uint batchIdx = gid / (seqLength * modelDim);
    uint seqIdx   = (gid / modelDim) % seqLength;
    uint dimIdx   = gid % modelDim;

    // Retrieve the propagated error from outputErrors explicitly
    float dOutput = outputErrors[batchIdx * seqLength * modelDim + seqIdx * modelDim + dimIdx];

    // Gradients w.r.t. weightsO and inputErrors
    for (uint i = 0; i < inputDim; ++i) {
        float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
        float gradO = dOutput * inputVal;

        atomic_fetch_add_explicit(&gradWeightsO[dimIdx * inputDim + i], gradO, memory_order_relaxed);

        float propagatedError = dOutput * weightsO[dimIdx * inputDim + i];
        atomic_fetch_add_explicit(
            &inputErrors[batchIdx * seqLength * inputDim + seqIdx * inputDim + i],
            propagatedError,
            memory_order_relaxed
        );
    }

    float scale = sqrt(float(modelDim));

    // Compute Q explicitly for current position
    float Q = 0.0f;
    for (uint i = 0; i < inputDim; ++i) {
        float inputVal = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
        Q += inputVal * weightsQ[i * modelDim + dimIdx];
    }

    // Compute gradients for weightsQ, weightsK, weightsV
    for (uint j = 0; j < seqLength; ++j) {
        float K = 0.0f, V = 0.0f;
        for (uint i = 0; i < inputDim; ++i) {
            float inputVal = input[batchIdx * seqLength * inputDim + j * inputDim + i];
            K += inputVal * weightsK[i * modelDim + dimIdx];
            V += inputVal * weightsV[i * modelDim + dimIdx];
        }

        float score = (Q * K) / scale;
        float attentionWeight = exp(score);

        float dScore = attentionWeight * (V * dOutput) / scale;

        for (uint i = 0; i < inputDim; ++i) {
            float inputQ = input[batchIdx * seqLength * inputDim + seqIdx * inputDim + i];
            float inputK = input[batchIdx * seqLength * inputDim + j * inputDim + i];
            float inputV = inputK; // input for V and K share same input position

            atomic_fetch_add_explicit(
                &gradWeightsQ[i * modelDim + dimIdx],
                inputQ * dScore * K,
                memory_order_relaxed
            );

            atomic_fetch_add_explicit(
                &gradWeightsK[i * modelDim + dimIdx],
                inputK * dScore * Q,
                memory_order_relaxed
            );

            atomic_fetch_add_explicit(
                &gradWeightsV[i * modelDim + dimIdx],
                inputV * attentionWeight * dOutput,
                memory_order_relaxed
            );
        }
    }
}
