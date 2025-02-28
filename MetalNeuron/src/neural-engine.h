#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H

#include "data-source-manager.h"
#include "keyboard-controller.h"
#include "logger.h"
#include "input-layer.h"
#include "rnn-layer.h"
#include "dense-layer.h"
#include <functional>

namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class CompileOptions;
    class CommandBuffer;
}

class NeuralEngine {
public:
    NeuralEngine(MTL::Device* pDevice, int sequenceLength);
    ~NeuralEngine();

    void computeForward(std::function<void()> onComplete);
    void computeBackward(std::function<void()> onComplete);

    void computeLearnAndApplyUpdates(uint32_t iterations);
    void computeForwardIterations(uint32_t iterations);

    void keyPress(KeyPress* kp);
    void handleKeyStateChange();

    static constexpr int kMaxFramesInFlight = 3;

private:
    void buildComputePipeline();
    void buildBuffers();
    void shiftBuffers(); // New helper: shifts stored sequence buffers by one time step

    DataSourceManager* _pDataSourceManager;
    KeyboardController* _pKeyboardController;
    Logger* _pLogger;

    InputLayer* _pInputLayer;
    RNNLayer* _pRNNLayer;
    DenseLayer* _pDenseLayer;

    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pComputeLibrary;
    MTL::CompileOptions* _pCompileOptions;

    bool areBuffersBuilt;
    bool currentlyComputing;
    dispatch_semaphore_t _semaphore;

    int sequenceLength_;
    int globalTimestep; // Controls the time offset for generating new data
};

#endif // NEURAL_ENGINE_H
