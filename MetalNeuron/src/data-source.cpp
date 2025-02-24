//
//  height-map.cpp
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//

#include "data-source.h"
#include "math-lib.h"
#include <simd/simd.h>

#pragma mark - HeightMap
#pragma region HeightMap {

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
            float x  = rand() * 1.f;
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


#pragma endregion HeightMap }
