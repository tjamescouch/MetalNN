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
    //  Data Members: RNN–Based Multi-Layer Architecture
    // ---------------------------------------------------
    // Data sources for network activations and targets.
    DataSource          x;         // Input data
    DataSource          y_hat;     // Target output for the output layer

    // Weight matrices and bias vectors.
    // For the RNN hidden layer:
    DataSource          W1;        // Input-to–hidden weights (W_xh)
    DataSource          b1;        // Hidden layer biases
    // For the output layer:
    DataSource          W2;        // Hidden-to–output weights
    DataSource          b2;        // Output layer biases
    
    DataSource          rand1;
    DataSource          rand2;
    
    float plasticity1, plasticity2;

    // ---------------------------------------------------
    //  Metal Device, Command Queue, and Compute Pipeline States
    // ---------------------------------------------------
    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;

    MTL::Library*       _pComputeLibrary = nullptr;
    
    // Pipeline states for the RNN–based kernels:
    MTL::ComputePipelineState*    _pForwardRnnPipelineState = nullptr;     // For RNN hidden layer forward pass
    MTL::ComputePipelineState*    _pForwardOutputPipelineState = nullptr;  // For output layer forward pass
    MTL::ComputePipelineState*    _pLearnOutputPipelineState = nullptr;      // For output layer learning
    MTL::ComputePipelineState*    _pLearnRnnPipelineState = nullptr;         // For RNN layer learning

    // Function pointers for the RNN–based kernels:
    MTL::Function*                _pForwardRnnFn = nullptr;
    MTL::Function*                _pForwardOutputFn = nullptr;
    MTL::Function*                _pLearnOutputFn = nullptr;
    MTL::Function*                _pLearnRnnFn = nullptr;

    // ---------------------------------------------------
    //  Metal Buffers for the RNN–Based Network
    // ---------------------------------------------------
    // Buffers for activations and target.
    MTL::Buffer* _pBuffer_x       = nullptr; // Input data
    MTL::Buffer* _pBuffer_hidden  = nullptr; // Hidden layer activations (current)
    MTL::Buffer* _pBuffer_hidden_prev = nullptr; // Hidden state from previous timestep
    MTL::Buffer* _pBuffer_y       = nullptr; // Output layer activations
    MTL::Buffer* _pBuffer_y_hat   = nullptr; // Target output

    // Buffers for weights and biases for the RNN hidden layer.
    // W1 now serves as the input-to–hidden weight matrix.
    MTL::Buffer* _pBuffer_W1      = nullptr;
    MTL::Buffer* _pBuffer_b1      = nullptr;
    // Buffer for recurrent (hidden-to–hidden) weights.
    MTL::Buffer* _pBuffer_W_hh    = nullptr;
    
    // Buffers for weights and biases for the output layer.
    MTL::Buffer* _pBuffer_W2      = nullptr;
    MTL::Buffer* _pBuffer_b2      = nullptr;

    // Dimension buffers.
    // For the RNN hidden layer:
    MTL::Buffer* _pBuffer_M1      = nullptr; // Input dimension for hidden layer
    MTL::Buffer* _pBuffer_N1      = nullptr; // Hidden layer dimension
    // For the output layer:
    MTL::Buffer* _pBuffer_M2      = nullptr; // Hidden layer dimension (as input to output layer)
    MTL::Buffer* _pBuffer_N2      = nullptr; // Output layer dimension
    
    MTL::Buffer* _pBuffer_plasticity1 = nullptr;
    MTL::Buffer* _pBuffer_plasticity2 = nullptr;
    
    MTL::Buffer* _pBuffer_age1    = nullptr;
    MTL::Buffer* _pBuffer_age2    = nullptr;
    
    MTL::Buffer* _pBuffer_randomness1 = nullptr;
    MTL::Buffer* _pBuffer_randomness2 = nullptr;

    // Error buffers for backpropagation.
    MTL::Buffer* _pBuffer_error = nullptr;
    MTL::Buffer* _pBuffer_prev_error = nullptr;
    MTL::Buffer* _pBuffer_error_hidden = nullptr;
    MTL::Buffer* _pBuffer_prev_error_hidden = nullptr;

    // Accumulator buffers for gradient updates.
    // For the RNN hidden layer.
    MTL::Buffer* _pBuffer_WAccumulator1 = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator1 = nullptr;
    // For the output layer.
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
