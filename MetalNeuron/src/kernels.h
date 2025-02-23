//
//  shaders.h
//  LearnMetalCPP
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

kernel void add_arrays(
    device const float* inA, 
    device const float* inB, 
    device float* result, 
    uint id [[thread_position_in_grid]]) {
    result[id] = inA[id] + inB[id];
}
)";


}


#pragma endregion Declarations }
#endif
