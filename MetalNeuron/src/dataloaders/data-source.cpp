#include "data-source.h"
#include <random>
#include <cstdio>
#include "data-source.h"
#include <algorithm>
#include <cmath>
#include <cstdlib> // for rand()
#include <ctime>   // for srand()



// Random number generation setup
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> uniform_dist(-1.0, 1.0);

DataSource::DataSource(int width, int height, int sequenceLength)
    : width_(width), height_(height), sequenceLength_(sequenceLength),
      data_(sequenceLength, std::vector<float>(width * height, 0.0f)) {}

DataSource::~DataSource() {}

void DataSource::allocate_buffers() {
    // buffers are allocated in the constructor; no additional allocation needed here
}

void DataSource::shift_buffers() {
    size_t num_data = get_num_data();
    for (int t = 0; t < sequenceLength_ - 1; ++t) {
        float* current = get_data_buffer_at(t);
        float* next = get_data_buffer_at(t + 1);
        std::copy(next, next + num_data, current);
    }

    // Clear the last timestep
    float* last = get_data_buffer_at(sequenceLength_ - 1);
    std::fill(last, last + num_data, 0.0f);
}


void DataSource::randomize_buffers(double timeOffset) {
    srand(static_cast<unsigned>(time(nullptr) + timeOffset));
    for (int t = 0; t < sequenceLength_; ++t) {
        float* buffer = get_data_buffer_at(t);
        size_t num_data = get_num_data();
        for (int i = 0; i < num_data; ++i) {
            buffer[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

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
