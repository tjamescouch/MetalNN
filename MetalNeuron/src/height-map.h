//
//  height-map.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
#ifndef HEIGHT_MAP_H
#define HEIGHT_MAP_H

#pragma region Declarations {

#include "common.h"
#include <thread>

#include <array>

class HeightMap
{
public:
    HeightMap(int num_samples_per_row, float width);
    ~HeightMap();
    
    size_t get_num_vertices();
    size_t get_num_normals();
    size_t get_num_tangents();
    size_t get_num_bitangents();
    size_t get_num_colors();
    simd::float3* get_position_buffer();
    simd::float3* get_color_buffer();
    float get_height(float x, float y);
    
    void build();
    
    simd::float3* get_normal_buffer();
    simd::float3* get_tangent_buffer();
    simd::float3* get_bitangent_buffer();
    
    
    // Asynchronous build interface
    template<typename Callback>
    void buildAsync(Callback onComplete) {
        std::thread([this, onComplete]() {
            // 1) Do expensive build on background thread
            this->build();
            // 2) Inform caller that build is done
            onComplete();
        }).detach();
    }
    
private:
    std::vector<simd::float3> positions;
    std::vector<simd::float3> normals;
    std::vector<simd::float3> colors;
    std::vector<simd::float3> tangents;
    std::vector<simd::float3> bitangents;
    std::vector<simd::float2> gridUVs;
    
    int num_samples_per_row;
    float stride;
    float width;
    
};

#pragma endregion Declarations }
#endif

