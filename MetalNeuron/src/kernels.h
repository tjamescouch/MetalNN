/*
//  kernels.h
//
//  Created by James Couch on 2025-02-17.
*/

#ifndef KERNELS_H
#define KERNELS_H

#pragma region Declarations {

namespace kernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

const float learning_rate_w = 0.01f;
const float learning_rate_b = 0.001f;
const float plasticity = 1.f; 

inline float sigmoid(float x)
{
  return 1 / (1 + exp(-x));
}

inline float clamp_range(float x)
{
  return clamp(x, -1.f, 1.f);
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

inline float activationFunction(float x)
{
  return piecewise(x);
}

inline float expected(float in)
{
    return cos(10 * in);
}

kernel void forward(
    device const float* x               [[buffer(0)]],
    device       float* W               [[buffer(1)]],
    device       float* b               [[buffer(2)]],
    device       float* y               [[buffer(3)]],
    device       int* pM                [[buffer(4)]],
    device       int* pN                [[buffer(5)]],
    uint tid                            [[thread_position_in_grid]])
{
    int M = *pM;
    int N = *pN;
    
    if (tid >= N) return; 

    float sum = b[tid];
    for (uint i = 0; i < M; i++) {
        sum += x[i] * W[i * N + tid];
    }
    y[tid] = activationFunction(sum);
}

kernel void learn(
    device const float* x               [[buffer(0)]],
    device       float* W               [[buffer(1)]],
    device       float* b               [[buffer(2)]],
    device       float* y               [[buffer(3)]],
    device       int* pM                [[buffer(4)]],
    device       int* pN                [[buffer(5)]],
    device       float* W_accumulator   [[buffer(6)]],
    device       float* b_accumulator   [[buffer(6)]],
    uint tid                            [[thread_position_in_grid]])
{
    int M = *pM;
    int N = *pN;
    
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
