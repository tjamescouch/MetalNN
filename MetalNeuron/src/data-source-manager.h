#ifndef DATASOURCE_MANAGER_H
#define DATASOURCE_MANAGER_H

#include "data-source.h"
#include <functional>
#include <atomic>


class DataSourceManager {
public:
    DataSourceManager(int inputDim, int hiddenDim, int outputDim, int sequenceLength);
    ~DataSourceManager();

    DataSource x;      // Input data per timestep
    DataSource y;  // Target data per timestep

    void initialize(std::function<void()> onComplete,
                    double (*inputFunc)(double, double),
                    double (*targetFunc)(double, double));

    void buildInputAtTimestep(std::function<double(double, double)> inputFunc, int timestep, std::function<void()> onComplete);
    void buildTargetAtTimestep(std::function<double(double, double)> targetFunc, int timestep, std::function<void()> onComplete);
    std::atomic<int> completedTimesteps{0};

private:
    int sequenceLength_;
};

#endif // DATASOURCE_MANAGER_H
