#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

#include "common.h"
#include <thread>
#include <vector>
#include <functional>

class DataSource {
public:
    DataSource(int width, int height, int sequenceLength);
    ~DataSource();

    size_t get_num_data() const;
    float* get_data_buffer_at(int timestep);
    const float* get_data_buffer_at(int timestep) const;

    void buildAtTimestep(std::function<double(double, int)> f, int timestep);
    void initRandomAtTimestep(int timestep);

    template<typename Callback>
    void buildAsyncAtTimestep(std::function<double(double, int)> f, int timestep, Callback onComplete) {
        std::thread([this, f, timestep, onComplete]() {
            buildAtTimestep(f, timestep);
            onComplete();
        }).detach();
    }

    template<typename Callback>
    void initRandomAsyncAtTimestep(int timestep, Callback onComplete) {
        std::thread([this, timestep, onComplete]() {
            initRandomAtTimestep(timestep);
            onComplete();
        }).detach();
    }

private:
    int width_;
    int height_;
    int sequenceLength_;
    std::vector<std::vector<float>> data_;  // One data vector per timestep
};

#endif // DATA_SOURCE_H
