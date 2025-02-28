#include "data-source.h"
#include <random>
#include <cstdio>

// Random number generation setup
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);

DataSource::DataSource(int width, int height, int sequenceLength)
    : width_(width), height_(height), sequenceLength_(sequenceLength),
      data_(sequenceLength, std::vector<float>(width * height, 0.0f)) {}

DataSource::~DataSource() {}

size_t DataSource::get_num_data() const {
    return width_ * height_;
}

float* DataSource::get_data_buffer_at(int timestep) {
    return data_[timestep].data();
}

const float* DataSource::get_data_buffer_at(int timestep) const {
    return data_[timestep].data();
}

void DataSource::buildAtTimestep(std::function<double(double, int)> f, int timestep) {
    if (timestep < 0 || timestep >= sequenceLength_) return;

    printf("Generating data at timestep %d...\n", timestep);
    auto& timestepData = data_[timestep];
    for (int i = 0; i < width_ * height_; ++i) {
        timestepData[i] = static_cast<float>(f(i, timestep));
    }
    printf("Data generation finished for timestep %d. Generated %zu values\n", timestep, timestepData.size());
}

void DataSource::initRandomAtTimestep(int timestep) {
    if (timestep < 0 || timestep >= sequenceLength_) return;

    printf("Generating random data at timestep %d...\n", timestep);
    auto& timestepData = data_[timestep];
    for (auto& value : timestepData) {
        value = static_cast<float>(uniform_dist(gen));
    }
    printf("Random data generation finished for timestep %d. Generated %zu values\n", timestep, timestepData.size());
}
