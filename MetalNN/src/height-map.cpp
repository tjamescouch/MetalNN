//
//  height-map.cpp
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//

#include "height-map.h"
#include "math-lib.h"
#include <simd/simd.h>   // For simd::cross, simd::normalize

#pragma mark - HeightMap
#pragma region HeightMap {

HeightMap::HeightMap(int num_samples_per_row, float width)
{
    this->num_samples_per_row = num_samples_per_row;
    this->stride = width / num_samples_per_row;
    this->width = width;
}

HeightMap::~HeightMap()
{
}

simd::float3* HeightMap::get_position_buffer()
{
    return positions.data();
}

simd::float3* HeightMap::get_color_buffer()
{
    return colors.data();
}

simd::float3* HeightMap::get_normal_buffer()
{
    return normals.data();
}

simd::float3* HeightMap::get_tangent_buffer()
{
    return tangents.data();
}

simd::float3* HeightMap::get_bitangent_buffer()
{
    return bitangents.data();
}

size_t HeightMap::get_num_vertices()
{
    return positions.size();
}

size_t HeightMap::get_num_normals()
{
    return normals.size();
}

size_t HeightMap::get_num_tangents()
{
    return tangents.size();
}

size_t HeightMap::get_num_bitangents()
{
    return bitangents.size();
}

size_t HeightMap::get_num_colors()
{
    return colors.size();
}


// Helper: Return the fractional part of x.
static inline float fract(float x)
{
    return x - std::floor(x);
}

// Helper: Deterministic pseudo-random number generator.
// Given an integer seed, returns a float in [0,1).
static inline float pseudoRandom(int seed)
{
    return fract(std::sin(static_cast<float>(seed)) * 43758.5453f);
}

// Helper: Return a pseudo-random float in [minVal, maxVal] based on seed.
static inline float randomInRange(int seed, float minVal, float maxVal)
{
    float r = pseudoRandom(seed);
    return minVal + r * (maxVal - minVal);
}

float HeightMap::get_height(float x, float y)
{
    const float BASELINE_AMPLITUDE = 0.05f;
    const float BASELINE_FREQ_X = 0.3f;
    const float BASELINE_FREQ_Y = 0.3f;
    const float MOUNTAIN_DENSITY = 0.05f;

    // Baseline
    float baseline = BASELINE_AMPLITUDE * std::sin(BASELINE_FREQ_X * x) * std::cos(BASELINE_FREQ_Y * y);

    // Procedural "mountains"
    const int   NUM_MOUNTAINS   = (int)(this->width * this->width * MOUNTAIN_DENSITY);
    const float MIN_AMPLITUDE   = -0.3f;
    const float MAX_AMPLITUDE   =  1.7f;
    const float MIN_CENTER      = -this->width / 2.0f;
    const float MAX_CENTER      =  this->width / 2.0f;
    const float MIN_SIGMA       =  0.5f;
    const float MAX_SIGMA       =  2.0f;
    const float MIN_AXIS        =  0.5f;
    const float MAX_AXIS        =  2.5f;
    const float MAX_SQUARED_DISTANCE = 64.0f;

    float mountainHeight = 0.0f;
    for (int i = 0; i < NUM_MOUNTAINS; i++) {
        int baseSeed = 100 * i;

        float amplitude  = randomInRange(baseSeed + 1, MIN_AMPLITUDE, MAX_AMPLITUDE);
        float centerX    = randomInRange(baseSeed + 2, MIN_CENTER,    MAX_CENTER);
        float centerY    = randomInRange(baseSeed + 3, MIN_CENTER,    MAX_CENTER);
        float sigma      = randomInRange(baseSeed + 4, MIN_SIGMA,     MAX_SIGMA);

        float orientation = randomInRange(baseSeed + 5, 0.0f, 2.0f * float(M_PI));
        float majorAxis   = randomInRange(baseSeed + 6, MIN_AXIS, MAX_AXIS);
        float minorAxis   = randomInRange(baseSeed + 7, MIN_AXIS, MAX_AXIS);

        float dx = x - centerX;
        float dy = y - centerY;
        if ((dx*dx + dy*dy) > MAX_SQUARED_DISTANCE) {
            continue;
        }

        // Rotate (dx, dy)
        float cosTheta = std::cos(orientation);
        float sinTheta = std::sin(orientation);
        float dxPrime =  dx * cosTheta + dy * sinTheta;
        float dyPrime = -dx * sinTheta + dy * cosTheta;

        float ellipseDist = (dxPrime*dxPrime)/(majorAxis*majorAxis)
                          + (dyPrime*dyPrime)/(minorAxis*minorAxis);

        float gaussian = amplitude * std::exp(-ellipseDist / (2.0f * sigma*sigma));
        mountainHeight += gaussian;
    }

    return baseline + mountainHeight;
}

void HeightMap::build()
{
    printf("Building height map\n");

    float half_width = this->width / 2.0f;
    
    // 1) Reserve space for the final expanded vertex arrays (positions, normals, etc.)
    //    We'll build them after we compute vertex normals.
    //    Each cell has 2 triangles => 6 vertices. For an N x N grid of cells, there are
    //    (N-1)*(N-1) cells if you treat num_samples_per_row as N. Adjust if needed.
    size_t numCellsPerRow = this->num_samples_per_row - 1;  // cells in each row
    size_t totalNumCells  = numCellsPerRow * numCellsPerRow;
    positions.reserve(totalNumCells * 6);
    colors.reserve(positions.capacity());
    normals.reserve(positions.capacity());

    // 2) Build a 2D array of all vertex positions (for each [ix, iy] sample).
    //    This lets us easily reference positions for the corners of any cell without
    //    calling get_height multiple times.
    std::vector<simd::float3> gridPositions(this->num_samples_per_row * this->num_samples_per_row);
    for (int ix = 0; ix < this->num_samples_per_row; ++ix)
    {
        printf("%f\n", 100.0*(float)ix / this->num_samples_per_row);
        float px = ix * stride - half_width;
        for(int iy = 0; iy < this->num_samples_per_row; ++iy)
        {
            float pz = iy * stride - half_width;
            float h  = this->get_height(px, pz);
            
            // Flattened 2D index: (ix * num_samples_per_row + iy)
            gridPositions[ix * this->num_samples_per_row + iy] = simd::float3{ px, h, pz };
        }
    }
    
    // 2A) Create a UV for each sample
    this->gridUVs.resize(this->num_samples_per_row * this->num_samples_per_row);

    for (int ix = 0; ix < this->num_samples_per_row; ++ix)
    {
        float u = float(ix) / float(this->num_samples_per_row - 1);
        for (int iy = 0; iy < this->num_samples_per_row; ++iy)
        {
            float v = float(iy) / float(this->num_samples_per_row - 1);
            int index = ix * this->num_samples_per_row + iy;
            this->gridUVs[index] = simd::float2{u, v};
        }
    }

    // 3) Allocate an array to accumulate normals at each vertex.
    //    We'll add each face's normal to its 3 corner vertices here.
    std::vector<simd::float3> accumulatedNormals(this->num_samples_per_row * this->num_samples_per_row,
                                                simd::float3{0.f, 0.f, 0.f});
    
    std::vector<simd::float3> accumulatedTangents(
        this->num_samples_per_row * this->num_samples_per_row,
        simd::float3{0.f, 0.f, 0.f});

    std::vector<simd::float3> accumulatedBitangents(
        this->num_samples_per_row * this->num_samples_per_row,
        simd::float3{0.f, 0.f, 0.f});
    
    auto computeTangentBitangent = [&](int i0, int i1, int i2) {
        // Positions
        simd::float3 p0 = gridPositions[i0];
        simd::float3 p1 = gridPositions[i1];
        simd::float3 p2 = gridPositions[i2];

        // UVs
        simd::float2 uv0 = this->gridUVs[i0];
        simd::float2 uv1 = this->gridUVs[i1];
        simd::float2 uv2 = this->gridUVs[i2];

        // Edges of the triangle
        simd::float3 e1 = p1 - p0;
        simd::float3 e2 = p2 - p0;

        // UV deltas
        simd::float2 dUV1 = uv1 - uv0;
        simd::float2 dUV2 = uv2 - uv0;

        float denom = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        if (fabs(denom) < 1e-8f) {
            // Degenerate UV or something; skip or fallback
            return;
        }
        float r = 1.0f / denom;

        simd::float3 T = (e1 * dUV2.y - e2 * dUV1.y) * r;
        simd::float3 B = (e2 * dUV1.x - e1 * dUV2.x) * r;

        // Accumulate
        accumulatedTangents[i0] += T;
        accumulatedTangents[i1] += T;
        accumulatedTangents[i2] += T;

        accumulatedBitangents[i0] += B;
        accumulatedBitangents[i1] += B;
        accumulatedBitangents[i2] += B;
    };


    // 4) Loop over each cell to compute the two triangle normals, then add them
    //    to the 3 corners of each triangle.
    //    Make sure to go to (num_samples_per_row - 1) so we don’t go out of bounds.
    for (int ix = 0; ix < numCellsPerRow; ++ix)
    {
        printf("%f\n", 100.0*(float)ix / numCellsPerRow);
        for (int iy = 0; iy < numCellsPerRow; ++iy)
        {
            // Indices for the four corners of this cell:
            int idx00 = ix * this->num_samples_per_row + iy;       // (ix,   iy)
            int idx01 = ix * this->num_samples_per_row + (iy+1);   // (ix,   iy+1)
            int idx10 = (ix+1) * this->num_samples_per_row + iy;   // (ix+1, iy)
            int idx11 = (ix+1) * this->num_samples_per_row + (iy+1);//(ix+1, iy+1)

            simd::float3 v00 = gridPositions[idx00];
            simd::float3 v01 = gridPositions[idx01];
            simd::float3 v10 = gridPositions[idx10];
            simd::float3 v11 = gridPositions[idx11];

            // ----- Triangle 1: v00 -> v01 -> v10 -----
            simd::float3 side1_1 = v01 - v00;
            simd::float3 side2_1 = v10 - v00;
            simd::float3 normalT1 = simd::normalize(simd::cross(side1_1, side2_1));

            // Accumulate this normal into the three corners of the first triangle
            accumulatedNormals[idx00] += normalT1;
            accumulatedNormals[idx01] += normalT1;
            accumulatedNormals[idx10] += normalT1;

            // ----- Triangle 2: v10 -> v01 -> v11 -----
            // (or v10 -> v11 -> v01, depending on your winding)
            simd::float3 side1_2 = v01 - v10;
            simd::float3 side2_2 = v11 - v10;
            simd::float3 normalT2 = simd::normalize(simd::cross(side1_2, side2_2));

            // Accumulate normal for second triangle
            accumulatedNormals[idx10] += normalT2;
            accumulatedNormals[idx01] += normalT2;
            accumulatedNormals[idx11] += normalT2;
            
            computeTangentBitangent(idx00, idx01, idx10);
            computeTangentBitangent(idx10, idx01, idx11);
        }
    }

    // 5) Normalize and Orthonormalize
    for (int i = 0; i < this->num_samples_per_row * this->num_samples_per_row; i++)
    {
        simd::float3 n = simd::normalize(accumulatedNormals[i]);
        simd::float3 t = accumulatedTangents[i];
        simd::float3 b = accumulatedBitangents[i];

        // Orthogonalize T against N
        float dt = simd::dot(n, t);
        t = t - dt * n;
        t = simd::normalize(t);

        // B can be derived from N x T if you prefer
        simd::float3 bCross = simd::cross(n, t);
        if (simd::length_squared(bCross) < 1e-10f) {
            // Degenerate, fallback to something
            bCross = simd::normalize(b);
        }

        accumulatedNormals[i] = n;
        accumulatedTangents[i]   = t;
        accumulatedBitangents[i] = bCross;
    }

    // 6) Finally, build out your actual GPU-bound vertex arrays (positions, colors, normals).
    //    We do another pass, referencing the per-vertex normal from accumulatedNormals.
    for (int ix = 0; ix < numCellsPerRow; ++ix)
    {
        for (int iy = 0; iy < numCellsPerRow; ++iy)
        {
            int idx00 = ix * this->num_samples_per_row + iy;
            int idx01 = ix * this->num_samples_per_row + (iy+1);
            int idx10 = (ix+1) * this->num_samples_per_row + iy;
            int idx11 = (ix+1) * this->num_samples_per_row + (iy+1);

            simd::float3 v00 = gridPositions[idx00];
            simd::float3 v01 = gridPositions[idx01];
            simd::float3 v10 = gridPositions[idx10];
            simd::float3 v11 = gridPositions[idx11];

            simd::float3 n00 = accumulatedNormals[idx00];
            simd::float3 n01 = accumulatedNormals[idx01];
            simd::float3 n10 = accumulatedNormals[idx10];
            simd::float3 n11 = accumulatedNormals[idx11];
            
            simd::float3 t00 = accumulatedTangents[idx00];
            simd::float3 t01 = accumulatedTangents[idx01];
            simd::float3 t10 = accumulatedTangents[idx10];
            simd::float3 t11 = accumulatedTangents[idx11];

            simd::float3 b00 = accumulatedBitangents[idx00];
            simd::float3 b01 = accumulatedBitangents[idx01];
            simd::float3 b10 = accumulatedBitangents[idx10];
            simd::float3 b11 = accumulatedBitangents[idx11];

            // ----- First triangle (v00, v01, v10) -----
            positions.push_back(v00);
            normals.push_back(n00);
            tangents.push_back(t00);
            bitangents.push_back(b00);
            colors.push_back(simd::float3{1.0f, 0.0f, 0.0f}); // example color

            positions.push_back(v01);
            normals.push_back(n01);
            tangents.push_back(t01);
            bitangents.push_back(b01);
            colors.push_back(simd::float3{1.0f, 1.0f, 0.0f});

            positions.push_back(v10);
            normals.push_back(n10);
            tangents.push_back(t10);
            bitangents.push_back(b10);
            colors.push_back(simd::float3{1.0f, 0.5f, 0.0f});

            // ----- Second triangle (v10, v01, v11) -----
            positions.push_back(v10);
            normals.push_back(n10);
            tangents.push_back(t10);
            bitangents.push_back(b10);
            colors.push_back(simd::float3{1.0f, 0.0f, 1.0f});

            positions.push_back(v01);
            normals.push_back(n01);
            tangents.push_back(t01);
            bitangents.push_back(b01);
            colors.push_back(simd::float3{1.0f, 1.0f, 1.0f});

            positions.push_back(v11);
            normals.push_back(n11);
            tangents.push_back(t11);
            bitangents.push_back(b11);
            colors.push_back(simd::float3{1.0f, 0.5f, 1.0f});
        }
    }
    
    


    printf("HeightMap build finished. Generated %zu vertices\n", positions.size());
}


#pragma endregion HeightMap }
