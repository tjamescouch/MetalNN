#ifndef NEURAL_ENGINE_H
#define NEURAL_ENGINE_H

#include "data-source.h"
#include "common.h"
#include "key-press.h"
#include "data-source-manager.h"
#include "keyboard-controller.h"
#include "logger.h"
#include <map>
#include <functional>

// Forward declarations for Metal classes
namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class Buffer;
    class ComputePipelineState;
    class Function;
    class CompileOptions;
    class ComputeCommandEncoder;
}

class NeuralEngine
{
public:
    // Constructor / Destructor
    NeuralEngine(MTL::Device* pDevice);
    ~NeuralEngine();

    void computeForward(std::function<void()> onComplete);
    void computeLearn(std::function<void()> cb);
    void computeLearnAndApplyUpdates(uint32_t iterations);
    void computeForwardIterations(uint32_t iterations);
    
    void extractAllResults(int iterations);
    // The logging methods are now handled by Logger.
    void keyPress(KeyPress* kp);
    void handleKeyStateChange();
    static constexpr int kMaxFramesInFlight = 3;
    
private:
    // Internal Build Methods
    void buildComputePipeline();
    void buildBuffers();
    
    // Data Members: RNN–Based Multi-Layer Architecture
    DataSourceManager* _pDataSourceManager = nullptr;

    // Metal Device, Command Queue, and Compute Pipeline States
    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;
    MTL::Library*       _pComputeLibrary = nullptr;
    
    // Pipeline states for the RNN–based kernels
    MTL::ComputePipelineState*    _pForwardRnnPipelineState = nullptr;
    MTL::ComputePipelineState*    _pForwardOutputPipelineState = nullptr;
    MTL::ComputePipelineState*    _pLearnOutputPipelineState = nullptr;
    MTL::ComputePipelineState*    _pLearnRnnPipelineState = nullptr;

    // Function pointers for the RNN–based kernels
    MTL::Function*                _pForwardRnnFn = nullptr;
    MTL::Function*                _pForwardOutputFn = nullptr;
    MTL::Function*                _pLearnOutputFn = nullptr;
    MTL::Function*                _pLearnRnnFn = nullptr;

    // Metal Buffers for the RNN–Based Network
    MTL::Buffer* _pBuffer_x       = nullptr;
    MTL::Buffer* _pBuffer_hidden  = nullptr;
    MTL::Buffer* _pBuffer_hidden_prev = nullptr;
    MTL::Buffer* _pBuffer_y       = nullptr;
    MTL::Buffer* _pBuffer_y_hat   = nullptr;
    MTL::Buffer* _pBuffer_W1      = nullptr;
    MTL::Buffer* _pBuffer_b1      = nullptr;
    MTL::Buffer* _pBuffer_W_hh    = nullptr;
    MTL::Buffer* _pBuffer_W2      = nullptr;
    MTL::Buffer* _pBuffer_b2      = nullptr;
    MTL::Buffer* _pBuffer_M1      = nullptr;
    MTL::Buffer* _pBuffer_N1      = nullptr;
    MTL::Buffer* _pBuffer_M2      = nullptr;
    MTL::Buffer* _pBuffer_N2      = nullptr;
    MTL::Buffer* _pBuffer_plasticity1 = nullptr;
    MTL::Buffer* _pBuffer_plasticity2 = nullptr;
    MTL::Buffer* _pBuffer_age1    = nullptr;
    MTL::Buffer* _pBuffer_age2    = nullptr;
    MTL::Buffer* _pBuffer_randomness1 = nullptr;
    MTL::Buffer* _pBuffer_randomness2 = nullptr;
    MTL::Buffer* _pBuffer_error = nullptr;
    MTL::Buffer* _pBuffer_prev_error = nullptr;
    MTL::Buffer* _pBuffer_error_hidden = nullptr;
    MTL::Buffer* _pBuffer_prev_error_hidden = nullptr;
    MTL::Buffer* _pBuffer_WAccumulator1 = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator1 = nullptr;
    MTL::Buffer* _pBuffer_WAccumulator2 = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator2 = nullptr;
    MTL::CompileOptions* _pCompileOptions = nullptr;

    // Frame / Synchronization
    bool                    areBuffersBuilt = false;
    bool                    currentlyComputing = false;
    int                     _frame = 0;
    dispatch_semaphore_t    _semaphore;

    // User Input: Keyboard functionality handled by a separate class.
    KeyboardController* _pKeyboardController = nullptr;

    // Logging functionality is now handled by a separate Logger.
    Logger* _pLogger = nullptr;
};

#endif // NEURAL_ENGINE_H
