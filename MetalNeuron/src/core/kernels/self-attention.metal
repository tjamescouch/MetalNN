#include <metal_stdlib>

#include "common.metal"

using namespace metal;

// Choose maximum dimensions so you don't overflow thread stack memory.
// You must ensure at runtime: seqLength <= MAX_SEQ_LENGTH && modelDim <= MAX_MODEL_DIM.
#define MAX_SEQ_LENGTH 1024
#define MAX_MODEL_DIM 1024

inline uint i2D(uint width, uint row, uint col) {
    return row * width + col;
}

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


#include <metal_stdlib>
using namespace metal;

// Tune for your GPU
#define THREADGROUP_SIZE 64
#define TILE_DIM         16

// Utility atomic add (Metal 3+ usually has atomic_float natively)
inline void atomicAdd(volatile device atomic_float* ptr, float val)
{
    atomic_fetch_add_explicit(ptr, val, memory_order_relaxed);
}

kernel void backward_self_attention(
    device const float* input                [[buffer(0)]],   // [batchSize * seqLength * inputDim]
    device const float* weightsQ             [[buffer(1)]],   // [inputDim * modelDim]
    device const float* weightsK             [[buffer(2)]],   // [inputDim * modelDim]
    device const float* weightsV             [[buffer(3)]],   // [inputDim * modelDim]
    device const float* weightsO             [[buffer(4)]],   // [modelDim * inputDim]

    device const float* bufferQ              [[buffer(5)]],   // [batchSize * seqLength * modelDim]
    device const float* bufferK              [[buffer(6)]],   // [batchSize * seqLength * modelDim]
    device const float* bufferV              [[buffer(7)]],   // [batchSize * seqLength * modelDim]
    device const float* attn_weights         [[buffer(8)]],   // [batchSize * seqLength * seqLength]

    device atomic_float* inputErrors         [[buffer(9)]],   // [batchSize * seqLength * inputDim]
    device const float* outputErrors         [[buffer(10)]],  // [batchSize * seqLength * inputDim]

    device atomic_float* gradWeightsQ        [[buffer(11)]],  // [inputDim * modelDim]
    device atomic_float* gradWeightsK        [[buffer(12)]],  // [inputDim * modelDim]
    device atomic_float* gradWeightsV        [[buffer(13)]],  // [inputDim * modelDim]
    device atomic_float* gradWeightsO        [[buffer(14)]],  // [modelDim * inputDim]

    constant uint& batchSize                 [[buffer(15)]],
    constant uint& seqLength                 [[buffer(16)]],
    constant uint& inputDim                  [[buffer(17)]],
    constant uint& modelDim                  [[buffer(18)]],

    // NEW scratch buffer to store large per-thread arrays
    device float* scratch                    [[buffer(19)]],

    uint tid                                 [[thread_position_in_threadgroup]],
    uint blockId                             [[threadgroup_position_in_grid]],
    uint threadsPerGroup                     [[threads_per_threadgroup]],
    uint gridSize                            [[threads_per_grid]]
)
{
    // Each thread corresponds to (b, s)
    uint gid = blockId * threadsPerGroup + tid;
    uint totalTokens = batchSize * seqLength;
    if (gid >= totalTokens) {
        return;
    }
    uint b = gid / seqLength;
    uint s = gid % seqLength;

    //-----------------------------------------
    // 1) Precompute offsets
    //-----------------------------------------
    uint inputOffset   = (b * seqLength + s) * inputDim;
    uint outErrOffset  = (b * seqLength + s) * inputDim;
    uint attnOffsetBS  = (b * seqLength * seqLength) + (s * seqLength);

    //-----------------------------------------
    // 2) Layout of the scratch space
    //    We'll store:
    //       [ modelDim ] dAttn
    //       [ modelDim ] attnVal
    //       [ seqLength ] dAttnW_raw
    //       [ seqLength ] dAttnW
    //       [ seqLength * modelDim ] dV_s
    //       [ seqLength * modelDim ] dK_s
    //       [ modelDim ] dQ
    //
    // So total = 2*modelDim + 2*seqLength + 2*(seqLength*modelDim) + modelDim
    //          = 3*modelDim + 2*seqLength + 2*(seqLength*modelDim).
    //-----------------------------------------
    const uint scratchPerThread = (3*modelDim) + (2*seqLength) + (2 * seqLength * modelDim);

    // Base offset in "scratch" for this thread
    uint baseOffset = gid * scratchPerThread;
    // We'll define pointers for each sub-block:
    device float* dAttn_d    = scratch + baseOffset;                       // size = modelDim
    device float* attnVal_d  = dAttn_d + modelDim;                         // size = modelDim
    device float* dAttnW_raw = attnVal_d + modelDim;                       // size = seqLength
    device float* dAttnW     = dAttnW_raw + seqLength;                     // size = seqLength
    device float* dV_s       = dAttnW + seqLength;                         // size = seqLength*modelDim
    device float* dK_s       = dV_s + (seqLength * modelDim);              // size = seqLength*modelDim
    device float* dQ         = dK_s + (seqLength * modelDim);              // size = modelDim

    //-----------------------------------------
    // 3) Read dOut for this token from global
    //    We'll store it in registers or a small
    //    local array, since inputDim might be
    //    smaller than, say, 256 or so.
    //-----------------------------------------
    thread float dOut_i[256];
    for (uint i = 0; i < inputDim; i++) {
        dOut_i[i] = outputErrors[outErrOffset + i];
    }

    //-----------------------------------------
    // 4) dAttn = dOut_i * weightsO^T => [modelDim]
    //-----------------------------------------
    for (uint d = 0; d < modelDim; d++) {
        float sumVal = 0.0f;
        for (uint i = 0; i < inputDim; i++) {
            sumVal += dOut_i[i] * weightsO[d * inputDim + i];
        }
        dAttn_d[d] = sumVal;
    }

    //-----------------------------------------
    // 5) Recompute attnVal_d = attn_output(b,s,:)
    //    = sum_j [ attn_weights(b,s,j) * V(b,j,:) ]
    //-----------------------------------------
    for (uint d = 0; d < modelDim; d++) {
        float sumVal = 0.0f;
        for (uint j = 0; j < seqLength; j++) {
            float aw = attn_weights[attnOffsetBS + j];
            uint vOff_j = (b * seqLength + j) * modelDim + d;
            sumVal += aw * bufferV[vOff_j];
        }
        attnVal_d[d] = sumVal;
    }

    //-----------------------------------------
    // 6) gradWeightsO = outer( attnVal_d, dOut_i )
    //    We'll do a tile-based accumulation in
    //    threadgroup memory to reduce atomic collisions.
    //-----------------------------------------
    threadgroup float partialGradO[TILE_DIM * TILE_DIM];
    for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
        for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
            // Zero tile
            for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                partialGradO[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate partial
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;   // row in tile
                uint li = localIdx % TILE_DIM;   // col in tile
                uint d_ = dStart + ld;
                uint i_ = iStart + li;
                if (d_ < modelDim && i_ < inputDim) {
                    float val = attnVal_d[d_] * dOut_i[i_];
                    partialGradO[localIdx] = val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Atomic add to global
            for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                uint ld = localIdx / TILE_DIM;
                uint li = localIdx % TILE_DIM;
                uint d_ = dStart + ld;
                uint i_ = iStart + li;
                if (d_ < modelDim && i_ < inputDim) {
                    float v = partialGradO[localIdx];
                    atomicAdd(&(gradWeightsO[d_ * inputDim + i_]), v);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    //-----------------------------------------
    // 7) dV_s(j,d) = attn_weights(b,s,j)* dAttn_d[d]
    //    for j in [0..seqLength], store in scratch
    //-----------------------------------------
    for (uint j = 0; j < seqLength; j++) {
        float aw = attn_weights[attnOffsetBS + j];
        for (uint d = 0; d < modelDim; d++) {
            dV_s[j*modelDim + d] = aw * dAttn_d[d];
        }
    }

    //-----------------------------------------
    // 8) dAttn_weights raw => dAttnW_raw[j] = dot(dAttn, V(b,j,:))
    //    Then apply the softmax derivative:
    //      dAttnW[j] = attn_weights(b,s,j) *
    //        ( dAttnW_raw[j] - sum_{k}(attn_weights(b,s,k)* dAttnW_raw[k]) )
    //-----------------------------------------
    float sumSoftmax = 0.0f;
    // First compute all dAttnW_raw
    for (uint j = 0; j < seqLength; j++) {
        float sumVal = 0.0f;
        uint vOff_j = (b * seqLength + j)*modelDim;
        for (uint d = 0; d < modelDim; d++) {
            sumVal += dAttn_d[d] * bufferV[vOff_j + d];
        }
        dAttnW_raw[j] = sumVal;
    }
    // Then sum for the softmax part
    for (uint j = 0; j < seqLength; j++) {
        sumSoftmax += attn_weights[attnOffsetBS + j] * dAttnW_raw[j];
    }
    // Apply final
    for (uint j = 0; j < seqLength; j++) {
        float aw = attn_weights[attnOffsetBS + j];
        dAttnW[j] = aw * (dAttnW_raw[j] - sumSoftmax);
    }

    //-----------------------------------------
    // 9) dQ + dK_s => from dAttnW
    //    scale = 1 / sqrt(modelDim)
    //    dQ[d] = sum_j [ dAttnW[j] * K(b,j,d) ] * scale
    //    dK_s[j,d] = dAttnW[j] * Q(b,s,d) * scale
    //-----------------------------------------
    float scale = 1.0f / sqrt((float)modelDim);

    // Zero out dQ
    for (uint d = 0; d < modelDim; d++) {
        dQ[d] = 0.0f;
    }
    // Zero out dK_s in scratch
    for (uint j = 0; j < seqLength; j++) {
        for (uint d = 0; d < modelDim; d++) {
            dK_s[j*modelDim + d] = 0.0f;
        }
    }

    // Accumulate
    for (uint j = 0; j < seqLength; j++) {
        float coef = dAttnW[j] * scale;
        // add to dQ
        uint kOff_j = (b * seqLength + j)*modelDim;
        for (uint d = 0; d < modelDim; d++) {
            dQ[d] += coef * bufferK[kOff_j + d];
        }
        // fill dK_s
        // Q(b,s,d) offset is (b*seqLength + s)*modelDim + d
        uint qOff_bs = (b * seqLength + s)*modelDim;
        for (uint d = 0; d < modelDim; d++) {
            dK_s[j*modelDim + d] = coef * bufferQ[qOff_bs + d];
        }
    }

    //-----------------------------------------
    // 10) Accumulate into inputErrors + gradWeights
    //     for Q, K, V
    //-----------------------------------------

    //-----------------------------------------
    // 10a) Q => dInput(b,s) from dQ, gradWeightsQ
    //-----------------------------------------
    {
        // read input(b,s,:)
        thread float inVal[256];
        for (uint i = 0; i < inputDim; i++) {
            inVal[i] = input[inputOffset + i];
        }

        // dInput(b,s,i) += sum_d( dQ[d] * weightsQ[i,d] )
        for (uint i = 0; i < inputDim; i++) {
            float sumVal = 0.0f;
            for (uint d = 0; d < modelDim; d++) {
                sumVal += dQ[d] * weightsQ[i*modelDim + d];
            }
            atomicAdd(&(inputErrors[inputOffset + i]), sumVal);
        }

        // gradWeightsQ(i,d) += inVal[i]* dQ[d]
        // tile-based in threadgroup memory
        threadgroup float partialGradQ[TILE_DIM*TILE_DIM];
        for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
            for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                // zero tile
                for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                    partialGradQ[idx] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // accumulate
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        float val = inVal[ii] * dQ[dd];
                        partialGradQ[localIdx] = val;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // atomic add
                for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                    uint ld = localIdx / TILE_DIM;
                    uint li = localIdx % TILE_DIM;
                    uint dd = dStart + ld;
                    uint ii = iStart + li;
                    if (dd < modelDim && ii < inputDim) {
                        float v = partialGradQ[localIdx];
                        atomicAdd(&(gradWeightsQ[ii*modelDim + dd]), v);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    //-----------------------------------------
    // 10b) K => for each j in [0..seqLength]
    //   inputErrors(b,j) plus gradWeightsK
    //-----------------------------------------
    {
        for (uint j = 0; j < seqLength; j++) {
            uint inOff_j = (b*seqLength + j)*inputDim;
            thread float inVal_j[256];
            for (uint i = 0; i < inputDim; i++) {
                inVal_j[i] = input[inOff_j + i];
            }

            // dInput(b,j,i) += sum_d( dK_s[j,d]*weightsK[i,d] )
            for (uint i = 0; i < inputDim; i++) {
                float sumVal = 0.0f;
                for (uint d = 0; d < modelDim; d++) {
                    sumVal += dK_s[j*modelDim + d]* weightsK[i*modelDim + d];
                }
                atomicAdd(&(inputErrors[inOff_j + i]), sumVal);
            }

            // gradWeightsK(i,d) += inVal_j[i]* dK_s[j,d]
            threadgroup float partialGradK[TILE_DIM*TILE_DIM];
            for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
                for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                    for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                        partialGradK[idx] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint dd = dStart + ld;
                        uint ii = iStart + li;
                        if (dd < modelDim && ii < inputDim) {
                            float val = inVal_j[ii]* dK_s[j*modelDim + dd];
                            partialGradK[localIdx] = val;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint dd = dStart + ld;
                        uint ii = iStart + li;
                        if (dd < modelDim && ii < inputDim) {
                            atomicAdd(&(gradWeightsK[ii*modelDim + dd]), partialGradK[localIdx]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    //-----------------------------------------
    // 10c) V => for each j in [0..seqLength]
    //   inputErrors(b,j) plus gradWeightsV
    //-----------------------------------------
    {
        for (uint j = 0; j < seqLength; j++) {
            uint inOff_j = (b*seqLength + j)*inputDim;
            thread float inVal_j[256];
            for (uint i = 0; i < inputDim; i++) {
                inVal_j[i] = input[inOff_j + i];
            }

            // dInput(b,j,i) += sum_d( dV_s[j*modelDim + d]*weightsV[i*modelDim + d] )
            for (uint i = 0; i < inputDim; i++) {
                float sumVal = 0.0f;
                for (uint d = 0; d < modelDim; d++) {
                    sumVal += dV_s[j*modelDim + d] * weightsV[i*modelDim + d];
                }
                atomicAdd(&(inputErrors[inOff_j + i]), sumVal);
            }

            // gradWeightsV(i,d) += inVal_j[i] * dV_s[j*modelDim + d]
            threadgroup float partialGradV[TILE_DIM*TILE_DIM];
            for (uint dStart = 0; dStart < modelDim; dStart += TILE_DIM) {
                for (uint iStart = 0; iStart < inputDim; iStart += TILE_DIM) {
                    for (uint idx = tid; idx < TILE_DIM*TILE_DIM; idx += threadsPerGroup) {
                        partialGradV[idx] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint dd = dStart + ld;
                        uint ii = iStart + li;
                        if (dd < modelDim && ii < inputDim) {
                            float val = inVal_j[ii] * dV_s[j*modelDim + dd];
                            partialGradV[localIdx] = val;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint localIdx = tid; localIdx < TILE_DIM*TILE_DIM; localIdx += threadsPerGroup) {
                        uint ld = localIdx / TILE_DIM;
                        uint li = localIdx % TILE_DIM;
                        uint dd = dStart + ld;
                        uint ii = iStart + li;
                        if (dd < modelDim && ii < inputDim) {
                            atomicAdd(&(gradWeightsV[ii*modelDim + dd]), partialGradV[localIdx]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // Done for this token (b,s).
}

