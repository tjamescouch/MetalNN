//
//  renderer.cpp
//  LearnMetalCPP
//
//  Created by James Couch on 2024-12-07.
//

#include <simd/simd.h>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "renderer.h"
#include "height-map.h"
#include "shaders.h"
#include "math-lib.h"

// stb_image for loading PNG/JPG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#pragma mark - Renderer
#pragma region Renderer {

const static inline simd::float3 s_skyboxVertices[36] = {
    // +X face
    {+1, -1, -1}, {+1, -1, +1}, {+1, +1, -1},
    {+1, +1, -1}, {+1, -1, +1}, {+1, +1, +1},
    // -X face
    {-1, -1, +1}, {-1, -1, -1}, {-1, +1, +1},
    {-1, +1, +1}, {-1, -1, -1}, {-1, +1, -1},
    // +Y face
    {-1, +1, -1}, {+1, +1, -1}, {-1, +1, +1},
    {-1, +1, +1}, {+1, +1, -1}, {+1, +1, +1},
    // -Y face
    {+1, -1, -1}, {-1, -1, -1}, {+1, -1, +1},
    {+1, -1, +1}, {-1, -1, -1}, {-1, -1, +1},
    // +Z face
    {+1, -1, +1}, {-1, -1, +1}, {+1, +1, +1},
    {+1, +1, +1}, {-1, -1, +1}, {-1, +1, +1},
    // -Z face
    {-1, -1, -1}, {+1, -1, -1}, {-1, +1, -1},
    {-1, +1, -1}, {+1, -1, -1}, {+1, +1, -1}
};

const simd::float3 cameraPos = {0, 10, 0};
const simd::float3 centerOfScene = {0.f, 0.f, 0.f};
const simd::float3 worldUp  = simd::normalize(simd::float3{0.f, 1.f, 0.f});
const simd::float3 lightPos = simd::float3{100, 100.0f, 0.0f};
const simd::float3 lightDir = simd::normalize(centerOfScene - lightPos);


const int HEIGHT_MAP_WIDTH = 300;
const int HEIGHT_MAP_SAMPLES_PER_ROW = 300;


Renderer::Renderer(MTL::Device* pDevice)
: _pDevice(pDevice->retain())
, areBuffersBuilt(false)
, _frame(0)
, heightMap(HEIGHT_MAP_SAMPLES_PER_ROW, HEIGHT_MAP_WIDTH)
, camera(cameraPos)
{
    this->keyState = {};
    
    _pCommandQueue = _pDevice->newCommandQueue();
    buildTerrainPipeline();
    buildFrameData();
    buildDepthState();
    buildShadowMapResources();
    buildShadowPipeline();
    
    // Build the height map on a background thread
    heightMap.buildAsync([this]() {
        dispatch_async(dispatch_get_main_queue(), ^{
            this->buildSky();
            this->buildSkyPipelineState();
            this->buildBuffers();
            areBuffersBuilt = true;
        });
    });
    
    _semaphore = dispatch_semaphore_create(Renderer::kMaxFramesInFlight);
}

Renderer::~Renderer()
{
    _pShaderLibrary->release();
    _pArgBuffer->release();
    _pVertexPositionsBuffer->release();
    _pVertexColorsBuffer->release();
    
    for (int i = 0; i < Renderer::kMaxFramesInFlight; ++i)
    {
        _pFrameData[i]->release();
    }
    
    if (_pPSO) _pPSO->release();
    if (_pSkyPipelineState) _pSkyPipelineState->release();
    if (_pSkyDepthState) _pSkyDepthState->release();
    if (_pShadowDepthTex) _pShadowDepthTex->release();
    if (_pShadowDepthPipelineState) _pShadowDepthPipelineState->release();
    if (_pShadowDepthStencilState) _pShadowDepthStencilState->release();
    
    _pCommandQueue->release();
    _pDevice->release();
    _pDepthStencilState->release();
}


void Renderer::buildTerrainPipeline()
{
    using NS::StringEncoding::UTF8StringEncoding;
    const char* shaderSrc = shaders::shaderSrc;
    
    NS::Error* pError = nullptr;
    MTL::Library* pLibrary = _pDevice->newLibrary(
                                                  NS::String::string(shaderSrc, UTF8StringEncoding),
                                                  nullptr,
                                                  &pError
                                                  );
    if (!pLibrary)
    {
        __builtin_printf("Terrain library error: %s\n", pError->localizedDescription()->utf8String());
        assert(false);
    }
    
    MTL::Function* pVertexFn = pLibrary->newFunction(NS::String::string("vertexMain", UTF8StringEncoding));
    MTL::Function* pFragFn   = pLibrary->newFunction(NS::String::string("fragmentMain", UTF8StringEncoding));
    
    MTL::RenderPipelineDescriptor* pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pDesc->setVertexFunction(pVertexFn);
    pDesc->setFragmentFunction(pFragFn);
    
    pDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    pDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    
    NS::Error* pErr2 = nullptr;
    _pPSO = _pDevice->newRenderPipelineState(pDesc, &pErr2);
    if (!_pPSO)
    {
        __builtin_printf("Terrain pipeline error: %s\n", pErr2->localizedDescription()->utf8String());
        assert(false);
    }
    
    pDesc->release();
    pVertexFn->release();
    pFragFn->release();
    _pShaderLibrary = pLibrary;
}

void Renderer::buildFrameData()
{
    for (int i = 0; i < Renderer::kMaxFramesInFlight; ++i)
    {
        _pFrameData[i] = _pDevice->newBuffer(sizeof(FrameData), MTL::ResourceStorageModeManaged);
    }
}

void Renderer::buildDepthState()
{
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    _pDepthStencilState = _pDevice->newDepthStencilState(depthDesc);
    depthDesc->release();
}

void Renderer::buildShadowMapResources()
{
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType2D);
    desc->setPixelFormat(MTL::PixelFormatDepth32Float);
    desc->setWidth(2*4096);
    desc->setHeight(2*4096);
    desc->setMipmapLevelCount(1);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    
    _pShadowDepthTex = _pDevice->newTexture(desc);
    desc->release();
}

void Renderer::buildShadowPipeline()
{
    using NS::StringEncoding::UTF8StringEncoding;
    const char* shadowSrc = shaders::shadowSrc;
    
    NS::Error* pError = nullptr;
    MTL::Library* pShadowLib = _pDevice->newLibrary(NS::String::string(shadowSrc, UTF8StringEncoding), nullptr, &pError);
    if (!pShadowLib)
    {
        __builtin_printf("Shadow library error: %s\n", pError->localizedDescription()->utf8String());
        assert(false);
    }
    
    MTL::Function* shadowVertFn = pShadowLib->newFunction(NS::String::string("shadowVertex", UTF8StringEncoding));
    
    MTL::RenderPipelineDescriptor* pShadowDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pShadowDesc->setVertexFunction(shadowVertFn);
    pShadowDesc->setFragmentFunction(nullptr);
    pShadowDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    pShadowDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatInvalid);
    
    NS::Error* pErr2 = nullptr;
    _pShadowDepthPipelineState = _pDevice->newRenderPipelineState(pShadowDesc, &pErr2);
    if (!_pShadowDepthPipelineState)
    {
        __builtin_printf("Shadow Depth pipeline error: %s\n", pErr2->localizedDescription()->utf8String());
        assert(false);
    }
    
    MTL::DepthStencilDescriptor* shadowDepthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    shadowDepthDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    shadowDepthDesc->setDepthWriteEnabled(true);
    _pShadowDepthStencilState = _pDevice->newDepthStencilState(shadowDepthDesc);
    shadowDepthDesc->release();
    
    pShadowDesc->release();
    shadowVertFn->release();
    pShadowLib->release();
}

void Renderer::buildSkyPipelineState()
{
    using NS::StringEncoding::UTF8StringEncoding;
    const char* skyShaderSrc = shaders::skyShaderSrc;
    
    NS::Error* pError = nullptr;
    MTL::Library* pSkyLibrary = _pDevice->newLibrary(NS::String::string(skyShaderSrc, UTF8StringEncoding), nullptr, &pError);
    if (!pSkyLibrary)
    {
        __builtin_printf("Sky shader error: %s\n", pError->localizedDescription()->utf8String());
        assert(false);
    }
    
    MTL::Function* skyVertFn = pSkyLibrary->newFunction(NS::String::string("skyVertex", UTF8StringEncoding));
    MTL::Function* skyFragFn = pSkyLibrary->newFunction(NS::String::string("skyFragment", UTF8StringEncoding));
    
    MTL::RenderPipelineDescriptor* pSkyDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pSkyDesc->setVertexFunction(skyVertFn);
    pSkyDesc->setFragmentFunction(skyFragFn);
    
    pSkyDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    pSkyDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    
    NS::Error* pSkyErr = nullptr;
    _pSkyPipelineState = _pDevice->newRenderPipelineState(pSkyDesc, &pSkyErr);
    if (!_pSkyPipelineState)
    {
        __builtin_printf("Sky pipeline error: %s\n", pSkyErr->localizedDescription()->utf8String());
        assert(false);
    }
    
    MTL::DepthStencilDescriptor* skyDepthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    skyDepthDesc->setDepthCompareFunction(MTL::CompareFunctionAlways);
    skyDepthDesc->setDepthWriteEnabled(false);
    _pSkyDepthState = _pDevice->newDepthStencilState(skyDepthDesc);
    skyDepthDesc->release();
    
    pSkyDesc->release();
    skyVertFn->release();
    skyFragFn->release();
    pSkyLibrary->release();
}

void Renderer::buildBuffers()
{
    const size_t numVertices = heightMap.get_num_vertices();
    const size_t numColors = heightMap.get_num_colors();
    const size_t numNormals  = heightMap.get_num_normals();
    const size_t numTangents  = heightMap.get_num_tangents();
    const size_t numBitangents  = heightMap.get_num_bitangents();
    
    const size_t positionsDataSize = numVertices * sizeof(simd::float3);
    const size_t colorDataSize     = numColors * sizeof(simd::float3);
    const size_t normalsDataSize = numNormals * sizeof(simd::float3);
    const size_t tangentsDataSize = numTangents * sizeof(simd::float3);
    const size_t bitangentsDataSize = numBitangents * sizeof(simd::float3);
    
    // 1) Create and fill buffers
    _pVertexPositionsBuffer = _pDevice->newBuffer(positionsDataSize, MTL::ResourceStorageModeManaged);
    _pVertexColorsBuffer    = _pDevice->newBuffer(colorDataSize,     MTL::ResourceStorageModeManaged);
    _pVertexNormalsBuffer = _pDevice->newBuffer(normalsDataSize, MTL::ResourceStorageModeManaged);
    _pVertexTangentsBuffer = _pDevice->newBuffer(tangentsDataSize, MTL::ResourceStorageModeManaged);
    _pVertexBitangentsBuffer = _pDevice->newBuffer(bitangentsDataSize, MTL::ResourceStorageModeManaged);
    
    std::memcpy(_pVertexPositionsBuffer->contents(), heightMap.get_position_buffer(), positionsDataSize);
    std::memcpy(_pVertexColorsBuffer->contents(),    heightMap.get_color_buffer(),    colorDataSize);
    std::memcpy(_pVertexNormalsBuffer->contents(), heightMap.get_normal_buffer(), normalsDataSize);
    std::memcpy(_pVertexTangentsBuffer->contents(), heightMap.get_tangent_buffer(), tangentsDataSize);
    std::memcpy(_pVertexBitangentsBuffer->contents(), heightMap.get_bitangent_buffer(), bitangentsDataSize);
    
    _pVertexPositionsBuffer->didModifyRange(NS::Range::Make(0, positionsDataSize));
    _pVertexColorsBuffer->didModifyRange(NS::Range::Make(0, colorDataSize));
    _pVertexNormalsBuffer->didModifyRange(NS::Range::Make(0, normalsDataSize));
    _pVertexTangentsBuffer->didModifyRange(NS::Range::Make(0, tangentsDataSize));
    _pVertexBitangentsBuffer->didModifyRange(NS::Range::Make(0, bitangentsDataSize));
    
    // 3) Set up the argument buffer with all 3 arrays (positions, colors, normals)
    MTL::Function* pVertexFn = _pShaderLibrary->newFunction(NS::String::string("vertexMain", NS::StringEncoding::UTF8StringEncoding));
    MTL::ArgumentEncoder* pArgEncoder = pVertexFn->newArgumentEncoder(0);
    _pArgBuffer = _pDevice->newBuffer(pArgEncoder->encodedLength(), MTL::ResourceStorageModeManaged);
    
    pArgEncoder->setArgumentBuffer(_pArgBuffer, 0);
    pArgEncoder->setBuffer(_pVertexPositionsBuffer, 0, 0);
    pArgEncoder->setBuffer(_pVertexColorsBuffer,    0, 1);
    pArgEncoder->setBuffer(_pVertexNormalsBuffer,   0, 2);
    pArgEncoder->setBuffer(_pVertexTangentsBuffer,   0, 3);
    pArgEncoder->setBuffer(_pVertexBitangentsBuffer,   0, 4);
    _pArgBuffer->didModifyRange(NS::Range::Make(0, _pArgBuffer->length()));
    
    pVertexFn->release();
    pArgEncoder->release();
}

void Renderer::buildSky()
{
    static const char* faceFilenames[6] = {
        "./px2.jpeg",
        "./nx2.jpeg",
        "./py2.jpeg",
        "./ny2.jpeg",
        "./pz2.jpeg",
        "./nz2.jpeg"
    };
    
    int firstW, firstH, comp;
    unsigned char* tmp = stbi_load(faceFilenames[0], &firstW, &firstH, &comp, 4);
    if (!tmp)
    {
        throw std::runtime_error(std::string("Failed to load image: ") + faceFilenames[0]);
    }
    stbi_image_free(tmp);
    
    MTL::TextureDescriptor* cubeDesc = MTL::TextureDescriptor::alloc()->init();
    cubeDesc->setTextureType(MTL::TextureTypeCube);
    cubeDesc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    cubeDesc->setWidth(firstW);
    cubeDesc->setHeight(firstH);
    
    uint32_t mipLevels = 1 + (uint32_t)std::floor(std::log2(std::max(firstW, firstH)));
    cubeDesc->setMipmapLevelCount(mipLevels);
    cubeDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
    
    _pSkyCubeTexture = _pDevice->newTexture(cubeDesc);
    cubeDesc->release();
    
    std::cout << "Created skyCubeTexture: "
    << _pSkyCubeTexture->width()
    << " x "
    << _pSkyCubeTexture->height()
    << std::endl;
    
    for (int face = 0; face < 6; face++)
    {
        int w, h, channels;
        unsigned char* facePixels = stbi_load(faceFilenames[face], &w, &h, &channels, 4);
        if (!facePixels)
        {
            throw std::runtime_error(std::string("Failed to load image: ") + faceFilenames[face]);
        }
        if (w != firstW || h != firstH)
        {
            stbi_image_free(facePixels);
            throw std::runtime_error("Skybox images have mismatched dimensions!");
        }
        
        MTL::Region region = MTL::Region::Make2D(0, 0, w, h);
        _pSkyCubeTexture->replaceRegion(region, 0, face, facePixels, w * 4, 0);
        stbi_image_free(facePixels);
    }
    
    size_t skyboxSize = sizeof(s_skyboxVertices);
    _pSkyboxVertexBuffer = _pDevice->newBuffer(skyboxSize, MTL::ResourceStorageModeManaged);
    std::memcpy(_pSkyboxVertexBuffer->contents(), s_skyboxVertices, skyboxSize);
    _pSkyboxVertexBuffer->didModifyRange(NS::Range::Make(0, skyboxSize));
    
    std::cout << "Skybox build finished.\n";
}

// ------------------------------------------------------
// DRAW
// ------------------------------------------------------
void Renderer::draw(MTK::View* pView)
{
    if (!areBuffersBuilt) return;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    _frame = (_frame + 1) % Renderer::kMaxFramesInFlight;
    
    // (1) Shadow pass
    simd::float4x4 lightViewProj = drawShadowPass();
    
    // (2) Main pass
    drawMainPass(pView, lightViewProj);
    
    pPool->release();
}


simd::float4x4 Renderer::drawShadowPass()
{
    // Create view matrix (light looking at the scene)
    simd::float4x4 lightView = mathlib::lookAtMatrix(lightPos, centerOfScene, worldUp);
    
    // Orthographic Projection Bounds
    float orthoSize = HEIGHT_MAP_WIDTH / 2.0;  // Adjust based on scene size
    float left   = -orthoSize;
    float right  = orthoSize;
    float bottom = -orthoSize;
    float top    = orthoSize;
    float nearZ  = 100.0f;
    float farZ   = 500.0f;
    
    
    // Create orthographic projection matrix
    simd::float4x4 lightProj = mathlib::makeOrthographicMatrix(left, right, bottom, top, nearZ, farZ);
    //simd::float4x4 lightProj = mathlib::makeProjectionMatrix(mathlib::radians(45), 1, nearZ, farZ);
    
    // Compute final light-space matrix
    simd::float4x4 lightViewProj = simd_mul(lightProj, lightView);
    
    // Render pass
    MTL::RenderPassDescriptor* rpd = MTL::RenderPassDescriptor::alloc()->init();
    rpd->depthAttachment()->setTexture(_pShadowDepthTex);
    rpd->depthAttachment()->setLoadAction(MTL::LoadActionClear);
    rpd->depthAttachment()->setStoreAction(MTL::StoreActionStore);
    rpd->depthAttachment()->setClearDepth(1.0);
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    MTL::RenderCommandEncoder* enc = cmdBuf->renderCommandEncoder(rpd);
    
    enc->setRenderPipelineState(_pShadowDepthPipelineState);
    enc->setDepthStencilState(_pShadowDepthStencilState);
    enc->setDepthBias(2.0f, 4.0f, 0.01f);
    enc->setCullMode(MTL::CullModeNone);
    
    // Provide light matrix to shaders
    ShadowUniforms su;
    su.lightViewProjMatrix = lightViewProj;
    enc->setVertexBytes(&su, sizeof(su), 1);
    
    // Draw terrain positions
    enc->setVertexBuffer(_pVertexPositionsBuffer, /*offset*/0, /*index*/0);
    
    size_t vertexCount = heightMap.get_num_vertices();
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), vertexCount);
    
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
    
    rpd->release();
    return lightViewProj;
}



void Renderer::drawMainPass(MTK::View* pView, simd::float4x4 lightViewProj)
{
    MTL::RenderPassDescriptor* rpd = pView->currentRenderPassDescriptor();
    if (!rpd) return;
    
    auto depthAttch = rpd->depthAttachment();
    depthAttch->setTexture(pView->depthStencilTexture());
    depthAttch->setLoadAction(MTL::LoadActionClear);
    depthAttch->setStoreAction(MTL::StoreActionStore);
    depthAttch->setClearDepth(1.0f);
    
    float dt = 1.0f / 60.0f;
    float moveSpeed = 2.0f;
    float rotationSpeed  = 10.5f;
    float moveAmount = moveSpeed * dt;
    
    if (this->keyState[26]) { camera.moveForward(-moveAmount); }  // W
    if (this->keyState[22]) { camera.moveBackward(-moveAmount); } // S
    if (this->keyState[4])  { camera.moveRight(moveAmount);   }   // A
    if (this->keyState[7])  { camera.moveLeft(moveAmount);    }   // D
    if (this->keyState[8])  { camera.moveUp(moveAmount);      }   // E
    if (this->keyState[20]) { camera.moveDown(moveAmount);    }   // Q
    
    // Example rotation
    float pitchAmount = rotationSpeed * dt;
    float yawAmount   = rotationSpeed * dt;
    float rollAmount  = rotationSpeed * dt;
    
    if (this->keyState[29]) { camera.yaw(yawAmount); }     // Z
    if (this->keyState[6])  { camera.yaw(-yawAmount); }    // C
    if (this->keyState[82]) { camera.pitch(pitchAmount); } // Up
    if (this->keyState[81]) { camera.pitch(-pitchAmount); }// Down
    if (this->keyState[80]) { camera.roll(rollAmount); }   // Left
    if (this->keyState[79]) { camera.roll(-rollAmount); }  // Right
    
    // Update per-frame data
    MTL::Buffer* pFrameDataBuffer = _pFrameData[_frame];
    auto* fd = reinterpret_cast<FrameData*>(pFrameDataBuffer->contents());
    
    fd->viewMatrix       = camera.getViewMatrix();
    fd->cameraPosition   = camera.getPosition();
    fd->lightViewProjMatrix = lightViewProj;
    fd->lightDir = lightDir;
    pFrameDataBuffer->didModifyRange(NS::Range::Make(0, sizeof(FrameData)));
    
    MTL::CommandBuffer* cmdBuf = _pCommandQueue->commandBuffer();
    dispatch_semaphore_wait(_semaphore, DISPATCH_TIME_FOREVER);
    
    Renderer* self = this;
    cmdBuf->addCompletedHandler(^void(MTL::CommandBuffer* cb){
        dispatch_semaphore_signal(self->_semaphore);
    });
    
    MTL::RenderCommandEncoder* enc = cmdBuf->renderCommandEncoder(rpd);
    
    // (A) Draw sky
    drawSkybox(enc);
    
    // (B) Draw terrain
    drawTerrain(enc);
    
    enc->endEncoding();
    cmdBuf->presentDrawable(pView->currentDrawable());
    cmdBuf->commit();
}


void Renderer::drawSkybox(MTL::RenderCommandEncoder* enc)
{
    enc->setRenderPipelineState(_pSkyPipelineState);
    enc->setDepthStencilState(_pSkyDepthState);
    enc->setCullMode(MTL::CullModeFront);
    
    simd::float4x4 vmat = camera.getViewMatrix();
    SkyUniforms skyU;
    skyU.viewMatrix = vmat;
    skyU.viewMatrix.columns[3] = simd_make_float4(0,0,0,1);
    
    enc->setVertexBytes(&skyU, sizeof(skyU), 1);
    enc->setVertexBuffer(_pSkyboxVertexBuffer, 0, 0);
    
    enc->setFragmentTexture(_pSkyCubeTexture, 0);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), 36);
}


void Renderer::drawTerrain(MTL::RenderCommandEncoder* enc)
{
    enc->setRenderPipelineState(_pPSO);
    enc->setDepthStencilState(_pDepthStencilState);
    enc->setCullMode(MTL::CullModeBack);
    enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
    
    // Provide the argument buffer with positions/colors/normals
    enc->setVertexBuffer(_pArgBuffer, /*offset*/0, /*index*/0);
    
    // Provide frame data in slot 1
    enc->setVertexBuffer(_pFrameData[_frame], /*offset*/0, /*index*/1);
    
    // Provide the same frame data to the fragment stage
    enc->setFragmentBuffer(_pFrameData[_frame], 0, 0);
    
    // Provide the shadow texture
    enc->setFragmentTexture(_pShadowDepthTex, 1);
    
    // Provide a sampler for reading that texture
    MTL::SamplerDescriptor* desc = MTL::SamplerDescriptor::alloc()->init();
    desc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    desc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    
    MTL::SamplerState* shadowSampler = _pDevice->newSamplerState(desc);
    desc->release();
    
    enc->setFragmentSamplerState(shadowSampler, 0.0f, FLT_MAX, 1);
    shadowSampler->release();
    
    // If needed, mark resource usage
    enc->useResource(_pVertexPositionsBuffer, MTL::ResourceUsageRead);
    enc->useResource(_pVertexColorsBuffer,    MTL::ResourceUsageRead);
    enc->useResource(_pVertexNormalsBuffer,   MTL::ResourceUsageRead);
    enc->useResource(_pVertexTangentsBuffer,   MTL::ResourceUsageRead);
    enc->useResource(_pVertexBitangentsBuffer,   MTL::ResourceUsageRead);
    
    // Draw
    size_t vertexCount = heightMap.get_num_vertices();
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), vertexCount);
}


void Renderer::keyPress(KeyPress* kp)
{
    if (kp->pressed)
    {
        keyState[kp->code] = true;
    }
    else
    {
        keyState.erase(kp->code);
    }
}

#pragma endregion Renderer
