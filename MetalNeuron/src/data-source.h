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
    DataSource(int width, int height);
    ~DataSource();
    
    size_t get_num_data();
    
    int get_width();
    int get_height();
    
    float* get_data_buffer();
    float get_data(float x, float y);
    
    void build();
    void initRandom();
    
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
    
    // Asynchronous build interface
    template<typename Callback>
    void initRandomAsync(Callback onComplete) {
        std::thread([this, onComplete]() {
            // 1) Do expensive build on background thread
            this->initRandom();
            // 2) Inform caller that build is done
            onComplete();
        }).detach();
    }
    
private:
    int width;
    int height;
    std::vector<float> data;
};

#pragma endregion Declarations }
#endif

