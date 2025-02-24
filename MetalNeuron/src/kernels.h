/*
//  kernels.h
//
//  Created by James Couch on 2025-02-17.
*/

#ifndef KERNELS_H
#define KERNELS_H

#pragma region Declarations {

namespace kernels {

const inline char* addArrayKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

inline float sigmoid(float x)
{
  return 1 / (1 + exp(-x));
}

inline float tanh_d(float d)
{
  return tanh(d);
}

inline float clamp_range(float d)
{
  return clamp(d, -1, 1);
}

inline float piecewise(float input)
{
  if (input > 1)
  {
    return 1;
  }
  else if (input > -1)
  {
    return input;
  }
  return -1;
}

inline float activationFunction(float d)
{
  return piecewise(d);
}

kernel void forward(
    device const simd::float3* x        [[buffer(0)]],
    device const simd::float3* W        [[buffer(1)]],
    device const simd::float3* b        [[buffer(2)]],
    device       simd::float3* y        [[buffer(3)]],
    device       simd::uint2*  W_dim    [[buffer(4)]],
    uint tid                            [[thread_position_in_grid]])
{
    uint M = W_dim.x;
    uint N = W_dim.y;

    if (tid >= N) return; 

    float sum = b[tid];
    for (uint i = 0; i < M; i++) {
        sum += x[i] * W[i * N + tid];
    }
    y[tid] = activationFunction(sum);
}
)";

} // namespace kernels

#pragma endregion Declarations }
#endif
