//
//  renderer.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
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
    void compute();
    void keyPress(KeyPress* kp);
    static constexpr int kMaxFramesInFlight = 3;
    
private:
    // ---------------------------------------------------
    //  Internal Build Methods
    // ---------------------------------------------------
    void buildComputePipeline();
    void buildBuffers();
    
    void doMainPass(MTK::View* pView);
    void doCompute(MTL::ComputeCommandEncoder* pEnc);
    
    // ---------------------------------------------------
    //  Data Members
    // ---------------------------------------------------
    DataSource          dataSource;
    
    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;
    
    // Pipeline states
    MTL::Library* _pComputeLibrary = nullptr;
    MTL::ComputePipelineState* _pComputePipelineState = nullptr;
    MTL::Function* _pComputeFn = nullptr;
    
    MTL::Buffer* _pInputBuffer = nullptr;
    MTL::Buffer* _pArgBuffer   = nullptr;
    
    
    MTL::CompileOptions* _pCompileOptions = nullptr;

    
    // Frame / Synch
    bool                    areBuffersBuilt     = false;
    int                     _frame              = 0;
    dispatch_semaphore_t    _semaphore;
    
    // Input
    std::map<long, bool>    keyState;
};

#endif // COMPUTER_H
