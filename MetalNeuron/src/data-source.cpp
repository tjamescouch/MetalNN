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

DataSource::DataSource(int num_samples_per_row, float width)
{
    this->num_samples_per_row = num_samples_per_row;
    this->stride = width / num_samples_per_row;
    this->width = width;
}

DataSource::~DataSource()
{
}

simd::float3* DataSource::get_data_buffer()
{
    return data.data();
}

size_t DataSource::get_num_data()
{
    return data.size();
}



float DataSource::get_data(float x, float y)
{
    const float z = x + y;
    
    return z;
}

void DataSource::build()
{
    printf("Generating data...\n");

    float half_width = this->width / 2.0f;
    

    size_t numCellsPerRow = this->num_samples_per_row - 1;  // cells in each row
    size_t totalNumCells  = numCellsPerRow * numCellsPerRow;
    data.reserve(totalNumCells * 6);

    for (int ix = 0; ix < this->num_samples_per_row; ++ix)
    {
        printf("%f\n", 100.0*(float)ix / this->num_samples_per_row);
        float px = ix * stride - half_width;
        for(int iy = 0; iy < this->num_samples_per_row; ++iy)
        {
            float pz = iy * stride - half_width;
            float py  = this->get_data(px, pz);
            
            this->data.push_back(simd::float3{px, py, pz});
        }
    }
    
    printf("Data generation finished. Generated %zu vectors\n", data.size());
}


#pragma endregion HeightMap }
