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

DataSource::DataSource(int num_samples_per_row)
{
    this->num_samples_per_row = num_samples_per_row;
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
    const float z = 7;
    
    return z;
}

void DataSource::build()
{
    printf("Generating data...\n");

    size_t numCellsPerRow = this->num_samples_per_row - 1;  // cells in each row
    size_t totalNumCells  = numCellsPerRow * numCellsPerRow;
    data.reserve(totalNumCells * 6);

    for (int ix = 0; ix < this->num_samples_per_row; ++ix)
    {
        printf("%f\n", 100.0*(float)ix / this->num_samples_per_row);
        for(int iy = 0; iy < this->num_samples_per_row; ++iy)
        {
            float iz  = this->get_data(ix, iy);
            
            this->data.push_back(simd::float3{(float)ix, (float)iy, iz});
        }
    }
    
    printf("Data generation finished. Generated %zu vectors\n", data.size());
}


#pragma endregion HeightMap }
