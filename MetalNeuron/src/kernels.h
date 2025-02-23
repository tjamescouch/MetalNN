//
//  kernels.h
//
//  Created by James Couch on 2025-02-17.
//

#ifndef KERNELS_H
#define KERNELS_H

#pragma region Declarations {

namespace kernels {

const inline char* addArrayKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct Buffers {
    device const float* inA  [[id(0)]];
    device const float* inB  [[id(1)]];
    device float*       result [[id(2)]];
};

kernel void add_arrays(constant Buffers& buffers [[buffer(0)]],
                       uint id [[thread_position_in_grid]])
{
    buffers.result[id] = 1.f;//buffers.inA[id] + buffers.inB[id];
}
)";

} // namespace kernels

#pragma endregion Declarations }
#endif
