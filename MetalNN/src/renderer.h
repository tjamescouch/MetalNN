//
//  renderer.h
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//
#ifndef RENDERER_H
#define RENDERER_H

#include "height-map.h"
#include "common.h"
#include "key-press.h"
#include "camera.h"
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


class Renderer
{
public:
    // Constructor / Destructor
    Renderer(MTL::Device* pDevice);
    ~Renderer();
    
    // The main draw call (invoked each frame)
    void draw(MTK::View* pView);
    
    // Handle keyboard / input events
    void keyPress(KeyPress* kp);
    
    // A constant for triple buffering
    static constexpr int kMaxFramesInFlight = 3;
    
private:
    // ---------------------------------------------------
    //  Internal Build Methods
    // ---------------------------------------------------
    void buildTerrainPipeline();       // Replaces "buildShaders()"
    void buildFrameData();
    void buildDepthState();
    void buildShadowMapResources();
    void buildShadowPipeline();
    void buildSky();
    void buildBuffers();
    void buildSkyPipelineState();
    
    // ---------------------------------------------------
    //  Internal Draw Methods
    // ---------------------------------------------------
    simd::float4x4 drawShadowPass();                   // The shadow pass
    void drawMainPass(MTK::View* pView, simd::float4x4 lightViewProj);     // The main color pass
    void drawSkybox(MTL::RenderCommandEncoder* pEnc);
    void drawTerrain(MTL::RenderCommandEncoder* pEnc);
    
    // ---------------------------------------------------
    //  Data Members
    // ---------------------------------------------------
    Camera              camera;
    HeightMap           heightMap;
    
    MTL::Device*        _pDevice;
    MTL::CommandQueue*  _pCommandQueue;
    
    // Pipeline states
    MTL::Library*             _pShaderLibrary      = nullptr;  // For terrain pipeline
    MTL::RenderPipelineState* _pPSO               = nullptr;   // Terrain pipeline
    MTL::RenderPipelineState* _pSkyPipelineState  = nullptr;   // Skybox pipeline
    MTL::RenderPipelineState* _pShadowDepthPipelineState = nullptr; // Shadow pass pipeline
    
    // Buffers for terrain
    MTL::Buffer* _pArgBuffer            = nullptr;
    MTL::Buffer* _pVertexPositionsBuffer= nullptr;
    MTL::Buffer* _pVertexColorsBuffer   = nullptr;
    MTL::Buffer* _pVertexNormalsBuffer = nullptr;
    MTL::Buffer* _pVertexTangentsBuffer = nullptr;
    MTL::Buffer* _pVertexBitangentsBuffer = nullptr;
    
    // Per-frame data (camera, transform, etc.)
    MTL::Buffer* _pFrameData[kMaxFramesInFlight];
    
    // Skybox
    MTL::Buffer*  _pSkyboxVertexBuffer = nullptr;
    MTL::Texture* _pSkyCubeTexture     = nullptr;
    
    // Shadow map
    MTL::Texture* _pShadowDepthTex     = nullptr;
    
    // Depth states
    MTL::DepthStencilState* _pDepthStencilState        = nullptr; // Normal depth for terrain
    MTL::DepthStencilState* _pSkyDepthState            = nullptr; // No-write depth for sky
    MTL::DepthStencilState* _pShadowDepthStencilState  = nullptr; // For the shadow pass
    
    // Frame / Synch
    bool                    areBuffersBuilt     = false;
    int                     _frame              = 0;
    dispatch_semaphore_t    _semaphore;
    
    // Input
    std::map<long, bool>    keyState;
};

#endif // RENDERER_H
