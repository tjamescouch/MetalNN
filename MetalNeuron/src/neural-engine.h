#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H
#include <vector>
#include <functional>

#include "model/model-config.h"
#include "data-source-manager.h"
#include "keyboard-controller.h"
#include "logger.h"
#include "layer.h"
#include "input-layer.h"
#include "rnn-layer.h"
#include "dense-layer.h"
#include "batch-normalization-layer.h"
#include "dataset.h"


namespace MTL {
class Device;
class CommandQueue;
class Library;
class CompileOptions;
class CommandBuffer;
class Buffer;
}

class NeuralEngine {
public:
    NeuralEngine(MTL::Device* pDevice, const ModelConfig& config);
    ~NeuralEngine();
    
    void runInference();
    
    void computeForward(std::function<void()> onComplete);
    void computeBackward(std::function<void()> onComplete);
    
    void computeBackwardIterations(uint32_t iterations);
    void computeForwardIterations(uint32_t iterations);
   // void createDynamicLayers(const ModelConfig& config);
    void connectDynamicLayers(const ModelConfig& config);
    void createDynamicLayers(const ModelConfig& config);

    void computeForwardSync();
    void computeBackwardSync();
    
    void keyPress(KeyPress* kp);
    void handleKeyStateChange();
    
    void initializeWithDataset(Dataset* dataset);
    
    
    static constexpr int kMaxFramesInFlight = 3;
    std::vector<Layer*> dynamicLayers_;
    
private:
    void buildComputePipeline();
    void buildBuffers();
    void shiftBuffers();
    
    DataSourceManager* _pDataSourceManager;
    KeyboardController* _pKeyboardController;
    Logger* _pLogger;
    
    InputLayer* _pInputLayer;
    
    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pComputeLibrary;
    MTL::CompileOptions* _pCompileOptions;
    
    MTL::Buffer* zeroBuffer_ = nullptr;
    
    bool areBuffersBuilt;
    bool currentlyComputing;
    dispatch_semaphore_t _semaphore;
    
    int globalTimestep; // Controls the time offset for generating new data
    int num_iterations = 1000;
    int input_dim  = 512;
    int hidden_dim = 512;
    int output_dim = 512;
};

#endif // NEURAL_ENGINE_H
