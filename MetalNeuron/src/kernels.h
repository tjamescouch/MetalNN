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

kernel void add_arrays(
    device const simd::float3* inA [[buffer(0)]],
    device const simd::float3* inB [[buffer(1)]],
    device simd::float3* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = float3(1.f,2.f,3.f);//inA[index] + inB[index];
}
)";

} // namespace kernels

#pragma endregion Declarations }
#endif
