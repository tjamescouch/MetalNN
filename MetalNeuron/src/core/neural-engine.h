#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H
#include <vector>
#include <functional>

#include "model-config.h"
#include "keyboard-controller.h"
#include "logger.h"
#include "layer.h"
#include "input-layer.h"
#include "rnn-layer.h"
#include "dense-layer.h"
#include "batch-normalization-layer.h"
#include "dataset.h"
#include "data-manager.h"


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
    NeuralEngine(MTL::Device* pDevice, const ModelConfig& config, DataManager* pDataManager);
    ~NeuralEngine();
    
    void runInference();
    
    void computeForward(std::function<void()> onComplete);
    void computeBackward(std::function<void()> onComplete);
    
    void computeBackwardIterations(uint32_t iterations);
    void computeForwardIterations(uint32_t iterations);
    void connectDynamicLayers(const ModelConfig& config);
    void createDynamicLayers(const ModelConfig& config);

    void computeForwardSync();
    void computeBackwardSync();
    
    void keyPress(KeyPress* kp);
    void handleKeyStateChange();
    
    void initializeWithDataset(Dataset* dataset);
    
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);
    
    
    static constexpr int kMaxFramesInFlight = 3;
    std::vector<Layer*> dynamicLayers_;
    
private:
    void buildComputePipeline();
    void buildBuffers();
    void shiftBuffers();
    
    KeyboardController* _pKeyboardController;
    Logger* _pLogger;
    DataManager* _pDataManager;
    
    InputLayer* _pInputLayer;
    
    MTL::Device* _pDevice;
    MTL::CommandQueue* _pCommandQueue;
    MTL::Library* _pComputeLibrary;
    MTL::CompileOptions* _pCompileOptions;
    
    MTL::Buffer* zeroBuffer_ = nullptr;
    
    bool areBuffersBuilt;
    bool currentlyComputing;
    dispatch_semaphore_t _semaphore;
    
    int batch_size;
    int epochs;
    //FIXME get from model configuration:
    int input_dim  = 512;
    int hidden_dim = 512;
    int output_dim = 512;
};

#endif // NEURAL_ENGINE_H
