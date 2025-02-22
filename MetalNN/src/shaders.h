//
//  shaders.h
//  LearnMetalCPP
//
//  Created by James Couch on 2025-02-17.
//

#ifndef SHADERS_H
#define SHADERS_H

#pragma region Declarations {

namespace shaders {

const inline char* shaderSrc = R"(#include <metal_stdlib>
using namespace metal;

constant float kPi = 3.14159265358979323846;

inline float radians(float degrees) {
    return degrees * (kPi / 180.0);
}

constant float fov = radians(65.0);

float4x4 makeProjectionMatrix(float fov, float aspect, float near, float far) {
    float yScale = 1.0 / tan(fov * 0.5);
    float xScale = yScale / aspect;
    float zScale = -(far + near) / (far - near);
    float zTranslation = -(2.0 * far * near) / (far - near);
    
    return float4x4(
        float4(xScale, 0.0,    0.0,     0.0),
        float4(0.0,    yScale, 0.0,     0.0),
        float4(0.0,    0.0,    zScale,  -1.0),
        float4(0.0,    0.0,    zTranslation, 0.0)
    );
}

float3x3 rotationMatrix(float3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;
    
    axis = normalize(axis); // Normalize the axis to ensure proper rotation
    
    return float3x3(
        float3(t * axis.x * axis.x + c,      t * axis.x * axis.y - s * axis.z,  t * axis.x * axis.z + s * axis.y),
        float3(t * axis.x * axis.y + s * axis.z,  t * axis.y * axis.y + c,      t * axis.y * axis.z - s * axis.x),
        float3(t * axis.x * axis.z - s * axis.y,  t * axis.y * axis.z + s * axis.x,  t * axis.z * axis.z + c)
    );
}

struct v2f {
    float4 position [[position]];
    float2 uv;
    float  height;
    half3  color;
    float3 normal;
    float3 worldPos;
    float3 tangent;
    float3 bitangent;
};

struct VertexData {
    device float3* positions [[id(0)]];
    device float3* colors    [[id(1)]];
    device float3* normals   [[id(2)]];
    device float3* tangents  [[id(3)]];
    device float3* bitangents[[id(4)]];
};

struct FrameData {
    float4x4 viewMatrix;
    float3 cameraPosition;
    float4x4 lightViewProjMatrix;
    float3 lightDir;
};

vertex v2f vertexMain(device const VertexData* vertexData [[buffer(0)]],
                     constant FrameData* frameData [[buffer(1)]],
                     uint vertexId [[vertex_id]])
{
    v2f out;

    float3 normal = vertexData->normals[vertexId];
    float3 pos = vertexData->positions[vertexId];
    float3 tangent = vertexData->tangents[vertexId];
    float3 bitangent = vertexData->bitangents[vertexId];

    float4 pos4 = float4(pos, 1.0);

    // Modified camera transform order
    float4x4 viewMatrix = frameData->viewMatrix;

    // Transform vertex position
    float4 viewPos = viewMatrix * pos4;
    
    // Use proper aspect ratio from your render target
    float aspect = 1.0;  // Adjust this to match your actual aspect ratio
    float near = 0.1;
    float far = 200.0;
    float4x4 projMatrix = makeProjectionMatrix(fov, aspect, near, far);
    
    out.position = projMatrix * viewPos;
    out.normal   = normal;
    out.tangent  = tangent;
    out.bitangent= bitangent;
    out.worldPos = pos;
    out.color = half3(vertexData->colors[vertexId]);
    out.height = pos.y;
    out.uv = pos.zx * 0.1;
    
    return out;
}

// ------------------------------------------------------------
// HELPER FUNCTIONS
// ------------------------------------------------------------
inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

inline float2 sigmoid(float2 x) {
    return 1 / (1 + exp(-x));
}

inline float3 sigmoid(float3 x) {
    return 1 / (1 + exp(-x));
}

inline float4 sigmoid(float4 x) {
    return 1 / (1 + exp(-x));
}

inline float random(float2 uv) {
    float noise = fract(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
    return noise;
}

inline float fade(float t) {
    return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

// A quick method to get a pseudo-random 2D gradient via `random()`:
inline float2 grad2D(int ix, int iy)
{
    // Convert integer coords to a float2 to feed our existing random()
    float2 seed = float2(ix, iy);
    float angle = random(seed) * 6.283185307f; // 2*pi
    return float2(cos(angle), sin(angle));
}

// Minimal 2D Perlin implementation
inline float perlinNoise(float2 P)
{
    // Integer cell coords
    float2 i = floor(P);
    // Fractional coords
    float2 f = P - i;

    // Smooth fade
    float2 u = float2(fade(f.x), fade(f.y));

    // Corner indices
    int ix = int(i.x);
    int iy = int(i.y);

    // Corner gradients
    float2 g00 = grad2D(ix,   iy);
    float2 g10 = grad2D(ix+1, iy);
    float2 g01 = grad2D(ix,   iy+1);
    float2 g11 = grad2D(ix+1, iy+1);

    // Relative positions from each corner
    float2 d00 = float2(f.x,     f.y);
    float2 d10 = float2(f.x-1.0, f.y);
    float2 d01 = float2(f.x,     f.y-1.0);
    float2 d11 = float2(f.x-1.0, f.y-1.0);

    // Dot products
    float dot00 = dot(g00, d00);
    float dot10 = dot(g10, d10);
    float dot01 = dot(g01, d01);
    float dot11 = dot(g11, d11);

    // Blend along x
    float lerpX1 = mix(dot00, dot10, u.x);
    float lerpX2 = mix(dot01, dot11, u.x);

    // Then along y
    float n = mix(lerpX1, lerpX2, u.y);

    // Result is roughly in [-1, +1]; you can scale as desired
    return n;
}

// 1) Compute the "strata" base color (sand, dirt, grass, snow)
inline float3 computeStrataColor(float height, float2 uvIn)
{
    const float randomnessIntensity = 0.01;
    float2 uv = float2(uvIn.x + randomnessIntensity * random(uvIn), uvIn.y + randomnessIntensity * random(uvIn));

    // Base levels
    const float baseSandLevel  = -0.1;
    const float baseDirtLevel  =  0.1;
    const float baseGrassLevel =  0.4;
    const float baseSnowLevel  =  1.9;
    
    // Warp frequencies/amplitudes
    const float warpFreq = 5.0;
    const float warpAmpSand  = 0.07;
    const float warpAmpDirt  = 0.08;
    const float warpAmpGrass = 0.2;
    const float warpAmpSnow  = 0.1;

    // Warped thresholds
    float warpSand  = sin(uv.x * warpFreq) * cos(uv.y * warpFreq) * warpAmpSand;
    float warpDirt  = sin(uv.x * (warpFreq + 1.0)) * cos(uv.y * (warpFreq + 1.0)) * warpAmpDirt;
    float warpGrass = sin(uv.x * (warpFreq + 2.0)) * cos(uv.y * (warpFreq + 2.0)) * warpAmpGrass;
    float warpSnow  = sin(uv.x * (warpFreq + 3.0)) * cos(uv.y * (warpFreq + 3.0)) * warpAmpSnow;
    
    float dynamicSandLevel  = baseSandLevel  + warpSand;
    float dynamicDirtLevel  = baseDirtLevel  + warpDirt;
    float dynamicGrassLevel = baseGrassLevel + warpGrass;
    float dynamicSnowLevel  = baseSnowLevel  + warpSnow;
    
    // Strata colors
    float3 sandColor  = float3(0.76, 0.70, 0.50);
    float3 dirtColor  = float3(0.4,  0.3,  0.1);
    float3 grassColor = float3(0.2,  0.4,  0.1);
    float3 snowColor  = float3(0.8,  0.75, 0.75);
    
    // Pick the base color depending on 'height'
    float3 strataColor;
    if (height < dynamicSandLevel) {
        strataColor = sandColor;
    }
    else if (height < dynamicDirtLevel) {
        float t = (height - dynamicSandLevel) / (dynamicDirtLevel - dynamicSandLevel);
        t = smoothstep(0.45, 0.55, t);
        strataColor = mix(sandColor, dirtColor, t);
    }
    else if (height < dynamicGrassLevel) {
        float t = (height - dynamicDirtLevel) / (dynamicGrassLevel - dynamicDirtLevel);
        t = smoothstep(0.1, 0.9, t);
        strataColor = mix(dirtColor, grassColor, t);
    }
    else if (height < dynamicSnowLevel) {
        float t = (height - dynamicGrassLevel) / (dynamicSnowLevel - dynamicGrassLevel);
        t = smoothstep(0.45, 0.55, t);
        strataColor = mix(grassColor, snowColor, t);
    }
    else {
        strataColor = snowColor;
    }
    
    return strataColor;
}

// 2) Apply a high-frequency variation / "noise" factor
inline float3 applyDetailNoise(float3 baseColor, float2 uv)
{
    const float BASE_FREQ1 = 100.0;
    const float BASE_FREQ2 = 55.0;
    const float BASE_FREQ3 = 80.0;
    const float BASE_FREQ4 = 40.0;
    const float BASE_SCALE = 0.1;
    
    float variation1 = sin(uv.x * BASE_FREQ1 + cos(uv.y * BASE_FREQ2));
    float variation2 = cos(uv.x * BASE_FREQ3 + sin(uv.y * BASE_FREQ4));
    float mixValue = (variation1 + variation2 + 2.0) / 4.0;

    float3 colorWithNoise = baseColor * (1.0 - BASE_SCALE)
                          + baseColor * (mixValue * BASE_SCALE);
    
    // Optional contrast tweak
    const float CONTRAST = 1.5;
    colorWithNoise = tanh(0.2*perlinNoise(30*uv) + (colorWithNoise - 0.5) * CONTRAST + 0.5);
    
    return colorWithNoise;
}

// 3) Add "Gaussian patches" or blotches
inline float3 applyGaussianPatches(float3 color, float2 uv)
{
    const float GAUSS_AMPLITUDE = 0.9;
    const float GAUSS_SIGMA     = 0.1;
    const float GAUSS_THRESHOLD = 0.05;
    
    const float2 PATCH_CENTER1 = float2(0.3, 0.7);
    const float2 PATCH_CENTER2 = float2(0.8, 0.2);
    
    float d1 = distance(uv, PATCH_CENTER1);
    float d2 = distance(uv, PATCH_CENTER2);
    
    float g1 = GAUSS_AMPLITUDE * exp(- (d1 * d1) / (2.0 * GAUSS_SIGMA * GAUSS_SIGMA));
    float g2 = GAUSS_AMPLITUDE * exp(- (d2 * d2) / (2.0 * GAUSS_SIGMA * GAUSS_SIGMA));
    
    if (g1 < GAUSS_THRESHOLD) { g1 = 0.0; }
    if (g2 < GAUSS_THRESHOLD) { g2 = 0.0; }
    
    float gaussianMix = clamp(g1 + g2, 0.0, 1.0);
    
    // Suppose we want to mix in a "dirt color" for these patches
    float3 patchColor = float3(0.4, 0.3, 0.1);
    float3 patchedColor = mix(color, patchColor, gaussianMix);
    
    return patchedColor;
}

// 4) Add random "grain" noise
inline float3 applyGrainNoise(float3 color, float2 uv)
{
    float noise = fract(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
    color += (noise - 0.5) * 0.05;
    return color;   
}

// 5) Distance Fog
//    This requires a fragment's world position and the camera position.
inline float3 applyDistanceFog(float3 color, float3 worldPos, float3 cameraPos)
{
    // Example fog parameters
    float3 skyColor = float3(0.8, 0.8, 0.85);
    float  fogStart = 90.0;
    float  fogEnd   = 150.0;
    
    // Distance from camera
    float dist = length(worldPos - cameraPos);
    
    // Smooth fade from fogStart to fogEnd
    float fogFactor = smoothstep(fogStart, fogEnd, dist);
    
    // Mix terrain color with sky/fog color
    float3 foggedColor = mix(color, skyColor, fogFactor);
    return foggedColor;
}

inline float sampleBumpHeight(float2 uv)
{
    const float BASE_FREQ1 = 40.0;
    const float BASE_FREQ2 = 25.0;
    const float BASE_FREQ3 = 10.0;
    const float BASE_FREQ4 = 4.0;
    
    float variation1 = sin(uv.x * BASE_FREQ1) + cos(uv.y * BASE_FREQ2);
    float variation2 = cos(uv.x * BASE_FREQ3) + sin(uv.y * BASE_FREQ4);
    float f = tanh(perlinNoise(200*uv) + (variation1*variation2) / 2.0);

    return f;   
}

// ------------------------------------------------------------
// FRAGMENT MAIN
// ------------------------------------------------------------
fragment half4 fragmentMain(
    v2f in [[stage_in]],
    constant FrameData* frameData [[buffer(0)]],
    texture2d<float> shadowTex    [[texture(1)]],
    sampler shadowSampler         [[sampler(1)]]
)
{
    //---------------------------------------------------------
    // (1) Build an orthonormal TBN
    //---------------------------------------------------------
    float3 N_geom = normalize(in.normal);
    float3 T_geom = normalize(in.tangent);
    float3 B_geom = normalize(in.bitangent);

    // Subtract from T any component along N to ensure perpendicular
    T_geom -= dot(T_geom, N_geom) * N_geom;
    T_geom  = normalize(T_geom);

    // Re-derive B so T, B, N form a perfect 3D basis
    B_geom  = normalize(cross(N_geom, T_geom));

    float3x3 TBN = float3x3(T_geom, B_geom, N_geom);

    //---------------------------------------------------------
    // (2) Procedural bump mapping via partial derivatives
    //---------------------------------------------------------
    float2 uv = in.uv;  // Must match how you generated tangents/bitangents
    float du = 0.01;
    float dv = 0.01;

    float h0 = sampleBumpHeight(uv);
    float hx = sampleBumpHeight(uv + float2(du, 0.0));
    float hy = sampleBumpHeight(uv + float2(0.0, dv));

    // partial derivatives of height wrt u and v
    float dhdu = (hx - h0) / du;
    float dhdv = (hy - h0) / dv;

    // Scale how strong the bumps are
    float bumpStrength = 0.009;
    float3 localNormal = float3(-dhdu * bumpStrength,
                                -dhdv * bumpStrength,
                                 1.0);
    localNormal = normalize(localNormal);

    // Transform local (tangent-space) normal into world space
    float3 N_bumped = normalize(TBN * localNormal);

    //---------------------------------------------------------
    // (3) Basic lighting
    //---------------------------------------------------------
    float3 cameraPos   = frameData->cameraPosition;
    float3 viewDir     = normalize(cameraPos - in.worldPos);
    float3 lightDir    = normalize(frameData->lightDir);
    float3 lightColor  = float3(0.95, 0.95, 0.9);
    float3 ambientColor= 0.4 * float3(0.5, 0.5, 0.6);

    float NdotL = max(dot(N_bumped, lightDir), 0.0);
    float3 diffuse = 0.8 * lightColor * NdotL;

    //---------------------------------------------------------
    // (4) Specular term (Blinn–Phong)
    //---------------------------------------------------------
    // Half vector is halfway between lightDir and viewDir
    float3 H = normalize(lightDir + viewDir);

    // Dot of normal and half vector
    float NdotH = max(dot(N_bumped, H), 0.0);

    // "Shininess" or "specular power" controls how sharp the highlight is
    const float shininess = 100.0;  // tweak to taste

    // The specular color can be white or tinted
    float3 specularColor = float3(1.0, 1.0, 1.0);

    // The Blinn–Phong specular term = (N·H)^shininess
    float3 specular = specularColor * pow(NdotH, shininess);

    // You can also scale how strong the spec is
    float specularIntensity = 0.5; // tweak to taste
    specular *= specularIntensity * lightColor;

    // --------------------------------------------------------
    // *** ADD A BASELINE OF PERLIN NOISE TO 'height' ***
    // --------------------------------------------------------
    float perlinVal  = perlinNoise(uv * 4.0);
    float newHeight  = in.height + 1.f * perlinVal;

    float3 strataColor = computeStrataColor(newHeight, uv);
    float3 detailColor = applyDetailNoise(strataColor, uv);
    float3 patchColor  = applyGaussianPatches(detailColor, uv);
    float3 noisyColor  = applyGrainNoise(patchColor, uv);

    // Example distance fog
    noisyColor         = applyDistanceFog(noisyColor, in.worldPos, cameraPos);

    float3 litColor    = noisyColor * (ambientColor + diffuse) + specular;

    //---------------------------------------------------------
    // (4) Shadow pass with the bumped normal
    //---------------------------------------------------------
    float4 lightClip = frameData->lightViewProjMatrix * float4(in.worldPos, 1.0);
    float3 lightNDC  = lightClip.xyz / lightClip.w;          // [-1..+1]
    float2 shadowUV  = 0.5 * (lightNDC.xy + float2(1.0, 1.0));// [0..1]
    float  fragDepth = 0.5 * (lightNDC.z + 1.0);

    float2 clampedUV = clamp(shadowUV, float2(0.0), float2(1.0));
    float  storedDepth = shadowTex.sample(shadowSampler, clampedUV).r;

    // Use the bumped normal for slope-based bias
    float slopeScale   = 0.01;
    float constantBias = 0.005;
    float slope        = 1.0 - dot(N_bumped, lightDir);
    float shadowBias   = max(0.005, constantBias + abs(slope) * slopeScale);

    bool inShadow = (fragDepth > (storedDepth + shadowBias));
    if (inShadow) {
        litColor = litColor * 0.1 + 0.2 * specular;
    }

    //---------------------------------------------------------
    // (5) Final output
    //---------------------------------------------------------
    return half4(half3(litColor), half(1.0));
}
)";



const inline char* skyShaderSrc = R"(
    #include <metal_stdlib>
    using namespace metal;
 
    constant float kPi = 3.14159265358979323846;
    inline float radians(float degrees) { return degrees * (kPi / 180.0); }
    constant float fov = radians(65.0);
    
    float4x4 makeProjectionMatrix(float fov, float aspect, float near, float far) {
        float yScale = 1.0 / tan(fov * 0.5);
        float xScale = yScale / aspect;
        float zScale = -(far + near) / (far - near);
        float zTranslation = -(2.0 * far * near) / (far - near);
        
        return float4x4(
            float4(xScale, 0.0, 0.0, 0.0),
            float4(0.0, yScale, 0.0, 0.0),
            float4(0.0, 0.0, zScale, -1.0),
            float4(0.0, 0.0, zTranslation, 0.0)
        );
    }
    
    // Vertex in/out
    struct SkyVertexIn {
        float3 position [[attribute(0)]];
    };
    struct SkyVertexOut {
        float4 position [[position]];
        float3 texCoord;
    };
    struct SkyUniforms {
        float4x4 viewMatrix;
    };
    
    vertex SkyVertexOut skyVertex(
        uint vertexId [[vertex_id]],
        device const float3* vertexPositions [[buffer(0)]],
        constant SkyUniforms& uniforms [[buffer(1)]]
    ) {
        SkyVertexOut out;
        
        float3 pos = vertexPositions[vertexId];
        
        // Use only the rotational part of the view matrix
        float4x4 rotationOnlyView = uniforms.viewMatrix;
        
        // Zero out the translation component
        rotationOnlyView.columns[3] = float4(0, 0, 0, 1);
        
        float4 worldPos = float4(pos, 1.0);
        float4 viewPos = rotationOnlyView * worldPos;  // Use original with just translation removed
        
        float aspect = 1.0;
        float near = 0.1;
        float far = 100.0;
        float4x4 projMatrix = makeProjectionMatrix(fov, aspect, near, far);
        
        out.position = projMatrix * viewPos;
        
        out.texCoord = pos; 
        return out;
    }

    fragment half4 skyFragment(
            SkyVertexOut in [[stage_in]],
            texturecube<float> skyTex [[texture(0)]]
    ) {
        float3 dir = normalize(in.texCoord);
        float4 color = skyTex.sample(sampler(address::clamp_to_edge), dir);
        return half4(color);
    }
)";


const inline char* shadowSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Must match your C++ struct layout
struct ShadowUniforms {
    float4x4 lightViewProjMatrix;
};

// Vertex function that transforms each vertex into the light's clip space.
// We'll bind the vertex buffer to index=0 and the ShadowUniforms to index=1.
vertex float4 shadowVertex(
    uint vertexID [[vertex_id]],
    device const float3* positions [[buffer(0)]],
    constant ShadowUniforms& uniforms [[buffer(1)]]
)
{
    float3 pos = positions[vertexID];
    float4 worldPos = float4(pos, 1.0);
    // Transform by the light's view-projection to get clip space
    float4 clipPos = uniforms.lightViewProjMatrix * worldPos;
    return clipPos;
}

// No fragment function is strictly needed if you only want to write depth.
// If your pipeline requires one, here's a trivial empty fragment:
fragment float shadowFragment() { return 1.0; }
)";

}

#pragma endregion Declarations }
#endif
