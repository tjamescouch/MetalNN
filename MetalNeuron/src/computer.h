#ifndef COMPUTER_H
#define COMPUTER_H

#include "data-source.h"
#include "common.h"
#include "key-press.h"
#include <map>

// Forward declarations for Metal classes
namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class RenderPipelineState;
    class Buffer;
    class Texture;
    class DepthStencilState;
    class ComputePipelineState;
    class Function;
    class CompileOptions;
    class ComputeCommandEncoder;
}

namespace MTK {
    class View;
}

struct FrameData
{
    simd::float4x4 viewMatrix;
    simd::float3 cameraPosition;
    simd::float4x4 lightViewProjMatrix;
    simd::float3 lightDir;
};

struct ShadowUniforms
{
    simd::float4x4 lightViewProjMatrix; // transforms world space → light clip space
};

struct SkyUniforms {
    simd::float4x4 viewMatrix;
};

class Computer
{
public:
    // Constructor / Destructor
    Computer(MTL::Device* pDevice);
    ~Computer();

    void computeForward(std::function<void()> onComplete);
    void computeLearn(std::function<void()> cb);
    void computeApplyUpdates(std::function<void()> cb);
    void computeLearnAndApplyUpdates(uint32_t iterations);
    void computeForwardIterations(uint32_t iterations);
    
    void extractAllResults(int iterations);
    void clearOutput();
    void logError();
    void logInformation(const std::string& filename, int remainingIterations);
    void keyPress(KeyPress* kp);
    void handleKeyStateChange();
    static constexpr int kMaxFramesInFlight = 3;
    
private:
    // ---------------------------------------------------
    //  Internal Build Methods
    // ---------------------------------------------------
    void buildComputePipeline();
    void buildBuffers();
    
    // ---------------------------------------------------
    //  Data Members: Multi-Layer Architecture
    // ---------------------------------------------------
    // Data sources for network activations and targets.
    DataSource          x;         // Input data
    DataSource          y_hat;     // Target output for the output layer

    // Weight matrices and bias vectors for each layer.
    DataSource          W1;        // Layer 1: Input -> Hidden weights
    DataSource          b1;        // Layer 1 biases
    DataSource          W2;        // Layer 2: Hidden -> Output weights
    DataSource          b2;        // Layer 2 biases
    
    DataSource          rand1;
    DataSource          rand2;  

    // ---------------------------------------------------
    //  Metal Device, Command Queue, and Compute Pipeline States
    // ---------------------------------------------------
    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;

    MTL::Library*                 _pComputeLibrary = nullptr;
    
    // Pipeline states for multi-layer kernels:
    MTL::ComputePipelineState*    _pForwardLayerPipelineState = nullptr;    // For Layer 1 (forward pass)
    MTL::ComputePipelineState*    _pForwardLayerPipelineState2 = nullptr;   // For Layer 2 (forward pass)
    MTL::ComputePipelineState*    _pLearnOutputPipelineState = nullptr;       // For output layer learning
    MTL::ComputePipelineState*    _pLearnHiddenPipelineState = nullptr;       // For hidden layer learning
    MTL::ComputePipelineState*    _pApplyUpdatesPipelineState = nullptr;      // Shared apply-updates kernel

    // Function pointers for multi-layer kernels:
    MTL::Function*                _pForwardLayerFn = nullptr;
    MTL::Function*                _pForwardLayerFn2 = nullptr;
    MTL::Function*                _pLearnOutputFn = nullptr;
    MTL::Function*                _pLearnHiddenFn = nullptr;
    MTL::Function*                _pApplyUpdatesFn = nullptr;

    // ---------------------------------------------------
    //  Metal Buffers for Multi-Layer Network
    // ---------------------------------------------------
    // Buffers for activations and target.
    MTL::Buffer* _pBuffer_x       = nullptr; // Input data
    MTL::Buffer* _pBuffer_hidden  = nullptr; // Hidden layer activations
    MTL::Buffer* _pBuffer_y       = nullptr; // Output layer activations
    MTL::Buffer* _pBuffer_y_hat   = nullptr; // Target output

    // Buffers for weights and biases for Layer 1.
    MTL::Buffer* _pBuffer_W1      = nullptr;
    MTL::Buffer* _pBuffer_b1      = nullptr;
    MTL::Buffer* _pBuffer_prev_W1      = nullptr;
    MTL::Buffer* _pBuffer_prev_b1      = nullptr;
    // Buffers for weights and biases for Layer 2.
    MTL::Buffer* _pBuffer_W2      = nullptr;
    MTL::Buffer* _pBuffer_b2      = nullptr;
    MTL::Buffer* _pBuffer_prev_W2      = nullptr;
    MTL::Buffer* _pBuffer_prev_b2      = nullptr;

    // Dimension buffers for Layer 1.
    MTL::Buffer* _pBuffer_M1      = nullptr; // Input dimension for Layer 1
    MTL::Buffer* _pBuffer_N1      = nullptr; // Hidden layer dimension (Layer 1 output)
    // Dimension buffers for Layer 2.
    MTL::Buffer* _pBuffer_M2      = nullptr; // Hidden layer dimension for Layer 2 input
    MTL::Buffer* _pBuffer_N2      = nullptr; // Output layer dimension
    MTL::Buffer* _pBuffer_randomness1 = nullptr;
    MTL::Buffer* _pBuffer_randomness2 = nullptr;

    // Error buffers for backpropagation.
    MTL::Buffer* _pBuffer_error = nullptr;
    MTL::Buffer* _pBuffer_prev_error = nullptr;
    MTL::Buffer* _pBuffer_error_hidden = nullptr;
    MTL::Buffer* _pBuffer_prev_error_hidden = nullptr;

    // Accumulator buffers for gradient updates.
    // For Layer 1.
    MTL::Buffer* _pBuffer_WAccumulator1 = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator1 = nullptr;
    // For Layer 2.
    MTL::Buffer* _pBuffer_WAccumulator2 = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator2 = nullptr;

    MTL::CompileOptions* _pCompileOptions = nullptr;

    // ---------------------------------------------------
    //  Frame / Synchronization
    // ---------------------------------------------------
    bool                    areBuffersBuilt = false;
    bool                    currentlyComputing = false;
    int                     _frame = 0;
    dispatch_semaphore_t    _semaphore;

    // ---------------------------------------------------
    //  User Input
    // ---------------------------------------------------
    std::map<long, bool>    keyState;
};

#endif // COMPUTER_H
