//
//  height-map.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#pragma region Declarations {

#include "common.h"
#include <thread>

#include <array>

class DataSource
{
public:
    DataSource(int num_samples_per_row);
    ~DataSource();
    
    size_t get_num_data();
    simd::float3* get_data_buffer();
    float get_data(float x, float y);
    
    void build();
    
    
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
    std::vector<simd::float3> data;
    
    int num_samples_per_row;
    
};

#pragma endregion Declarations }
#endif

