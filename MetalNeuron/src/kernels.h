/*
//  kernels.h
//
//  Updated by James Couch on 2025-02-24.
*/

#ifndef KERNELS_H
#define KERNELS_H

#pragma region Declarations {

namespace kernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

constant float learning_rate_w = 0.01f;
constant float learning_rate_b = 0.001f;
constant float plasticity = 1.f; 
constant float min_delta = 0.1f;
//constant float k_decay = 0.01f;
constant float max_de_dw = 0.5f;
constant float max_de_db = 0.1f;

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
  if (input > 1) return 1;
  else if (input > -1) return input;
  return -1;
}

inline float activationFunction(float x)
{
  return tanh(x);
}

kernel void forward(
    device const float* x               [[buffer(0)]],
    device       float* W               [[buffer(1)]],
    device       float* b               [[buffer(2)]],
    device       float* y               [[buffer(3)]],
    device       uint* pM               [[buffer(4)]],
    device       uint* pN               [[buffer(5)]],
    uint tid                            [[thread_position_in_grid]])
{
    uint M = *pM;
    uint N = *pN;
    
    if (tid >= N) return; 

    float sum = b[tid];
    for (uint i = 0; i < M; i++) {
        sum += x[i] * W[i * N + tid];
    }
    y[tid] = sum;//activationFunction(sum);
}

kernel void learn(
    device const float* x               [[buffer(0)]],
    device       float* W               [[buffer(1)]],
    device       float* b               [[buffer(2)]],
    device       float* y               [[buffer(3)]],
    device       float* error           [[buffer(4)]],
    device       uint* pM               [[buffer(5)]],
    device       uint* pN               [[buffer(6)]],
    device       float* W_accumulator   [[buffer(7)]],
    device       float* b_accumulator   [[buffer(8)]],
    uint tid                            [[thread_position_in_grid]])
{
    uint M = *pM;
    uint N = *pN;
    
    if (tid >= N) return; 

    float sum = b[tid];
    //for (uint i = 0; i < M; i++) {
    //    sum += x[i] * W[i * N + tid];
    //}
    y[tid] = sum;//activationFunction(sum);

    // Compute weight updates
    float delta_w, abs_delta_w, delta_error;
    for (uint i = 0; i < M; i++) {
        delta_w = W[i * N + tid] - W_accumulator[i * N + tid];
        abs_delta_w = fabs(delta_w);
        delta_error = error[tid];
        
        float delta_w_no_zero = abs_delta_w > min_delta ? delta_w : sign(delta_w) * min_delta;
        float de_dw = clamp(delta_error / delta_w_no_zero, -max_de_dw, max_de_dw);
        
        W_accumulator[i * N + tid] += learning_rate_w * plasticity * error[tid] * x[i] * de_dw * sign(de_dw);
        W_accumulator[i * N + tid] = clamp(W_accumulator[i * N + tid], -0.1f, 0.1f);
    }

    // Compute bias updates
    float delta_b = b[tid] - b_accumulator[tid];
    float abs_delta_b = fabs(delta_b);
    float delta_b_no_zero = abs_delta_b > min_delta ? delta_b : sign(delta_b) * min_delta;
    float de_db = clamp(error[tid] / delta_b_no_zero, -max_de_db, max_de_db);

    b_accumulator[tid] += learning_rate_b * plasticity * error[tid] * de_db * sign(de_db);
}

kernel void apply_updates(
    device float* W              [[buffer(0)]],
    device float* b              [[buffer(1)]],
    device float* W_accumulator  [[buffer(2)]],
    device float* b_accumulator  [[buffer(3)]],
    device int* pM               [[buffer(4)]],
    device int* pN               [[buffer(5)]],
    uint tid                     [[thread_position_in_grid]])
{
    uint M = *pM;
    uint N = *pN;
    
    if (tid >= N) return;
    
    for (uint i = 0; i < M; i++) {
        W[i * N + tid] += W_accumulator[i * N + tid];
        W_accumulator[i * N + tid] = 0;
    }
    
    b[tid] += b_accumulator[tid];
    b[tid] = clamp(b[tid], -1.f, 1.f);
    b_accumulator[tid] = 0;
}
)";

} // namespace kernels

#pragma endregion Declarations }
#endif
