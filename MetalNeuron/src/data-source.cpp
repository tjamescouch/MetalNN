//
//  height-map.cpp
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//

#include "data-source.h"
#include "math-lib.h"
#include <simd/simd.h>

#pragma mark - DataSource
#pragma region DataSource {

#include <random>

// Create a random number engine
std::random_device rd;
std::mt19937 gen(rd());

// Create a distribution that maps to [0, 1]
std::uniform_real_distribution<> uniform_0_to_1(0.0, 1.0);


DataSource::DataSource(int width, int height)
{
    this->width = width;
    this->height = height;
}

DataSource::~DataSource()
{
}

float* DataSource::get_data_buffer()
{
    return data.data();
}

size_t DataSource::get_num_data()
{
    return data.size();
}

float DataSource::get_data(float x, float y)
{
    const float z = 7;
    
    return z;
}

void DataSource::initRandom()
{
    printf("Generating data...\n");

    size_t numCellsPerRow = this->width - 1;  // cells in each row
    size_t totalNumCells  = numCellsPerRow * numCellsPerRow;
    data.reserve(totalNumCells * 6);

    for (int ix = 0; ix < this->width; ++ix)
    {
        for(int iy = 0; iy < this->height; ++iy)
        {
            float x  = uniform_0_to_1(gen);
            this->data.push_back(x);
        }
    }
    
    printf("Data generation finished. Generated %zu values\n", data.size());
}

void DataSource::build()
{
    printf("Generating data...\n");

    size_t numCellsPerRow = this->width - 1;  // cells in each row
    size_t totalNumCells  = numCellsPerRow * numCellsPerRow;
    data.reserve(totalNumCells * 6);

    for (int ix = 0; ix < this->width; ++ix)
    {
        for(int iy = 0; iy < this->height; ++iy)
        {
            float x  = sin(ix + iy * this->width);
            this->data.push_back(x);
        }
    }
    
    printf("Data generation finished. Generated %zu values\n", data.size());
}


#pragma endregion DataSource }
