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

    void computeForward();
    void computeLearn(std::function<void()> cb);
    void computeApplyUpdates(std::function<void()> cb);
    void computeLearnAndApplyUpdates(uint32_t iterations);
    
    void extractAllResults();
    void logInformation(const std::string& filename, MTL::Buffer* pBuffer_x, MTL::Buffer* pBuffer_y, MTL::Buffer* pBuffer_error);

    void extractResults(MTL::Buffer* pBuffer);
    void logInformation();
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
    //  Data Members
    // ---------------------------------------------------
    DataSource          x;
    DataSource          W;
    DataSource          b;

    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;

    // Pipeline states
    MTL::Library*                 _pComputeLibrary = nullptr;
    MTL::ComputePipelineState*    _pForwardComputePipelineState = nullptr;
    MTL::ComputePipelineState*    _pLearnComputePipelineState = nullptr;
    MTL::ComputePipelineState*    _pApplyUpdatesComputePipelineState = nullptr;
    
    MTL::Function*                _pForwardFn = nullptr;
    MTL::Function*                _pLearnFn = nullptr;
    MTL::Function*                _pApplyUpdatesFn = nullptr;

    // Buffers for the argument buffer approach
    MTL::Buffer* _pBuffer_x            = nullptr;
    MTL::Buffer* _pBuffer_W            = nullptr;
    MTL::Buffer* _pBuffer_b            = nullptr;
    MTL::Buffer* _pBuffer_y            = nullptr;
    MTL::Buffer* _pBuffer_M            = nullptr;
    MTL::Buffer* _pBuffer_N            = nullptr;
    MTL::Buffer* _pBuffer_error        = nullptr;
    MTL::Buffer* _pBuffer_WAccumulator = nullptr;
    MTL::Buffer* _pBuffer_bAccumulator = nullptr;

    MTL::CompileOptions* _pCompileOptions = nullptr;

    // Frame / Synchronization
    bool                    areBuffersBuilt = false;
    bool                    currentlyComputing = false;
    int                     _frame = 0;
    dispatch_semaphore_t    _semaphore;

    // User Input
    std::map<long, bool>    keyState;
};

#endif // COMPUTER_H
