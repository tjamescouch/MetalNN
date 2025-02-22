#include <simd/simd.h>
#include <cmath>

class Camera {
private:
    simd::float3 position;
    simd::float4x4 orientation;  // Stores overall rotation (no Euler angles)

public:
    float speed;
    float sensitivity;

    Camera(simd::float3 startPos)
    : position(startPos)
    , speed(1.0f)
    , sensitivity(2.0f)
    {
        // Start with identity orientation
        orientation = this->makeIdentity4x4();
    }

    simd::float3 getPosition() {
        return position;
    }
    
    // Move forward/back in local -Z or +Z
    void moveForward(float delta) {
        simd::float3 forward = getForwardVector();
        position += forward * delta * speed;
    }
    void moveBackward(float delta) {
        simd::float3 forward = getForwardVector();
        position -= forward * delta * speed;
    }

    // Move left/right in local X
    void moveRight(float delta) {
        simd::float3 right = getRightVector();
        position += right * delta * speed;
    }
    void moveLeft(float delta) {
        simd::float3 right = getRightVector();
        position -= right * delta * speed;
    }

    // Move up/down in local Y
    void moveUp(float delta) {
        simd::float3 up = getUpVector();
        position += up * delta * speed;
    }
    void moveDown(float delta) {
        simd::float3 up = getUpVector();
        position -= up * delta * speed;
    }

    // Rotate around the CAMERA’S LOCAL AXES
    void yaw(float degrees) {
        float rads = degrees * (M_PI / 180.0f) * sensitivity;
        // Yaw around the camera's local "up" axis
        simd::float3 localUp = getUpVector();
        simd::float4x4 r = makeRotation(localUp, rads);
        orientation = simd_mul(r, orientation);
    }

    void pitch(float degrees) {
        float rads = degrees * (M_PI / 180.0f) * sensitivity;
        // Pitch around the camera's local "right" axis
        simd::float3 localRight = getRightVector();
        simd::float4x4 r = makeRotation(localRight, rads);
        orientation = simd_mul(r, orientation);
    }

    void roll(float degrees) {
        float rads = degrees * (M_PI / 180.0f) * sensitivity;
        // Roll around the camera's local "forward" axis
        simd::float3 localForward = getForwardVector();
        simd::float4x4 r = makeRotation(localForward, rads);
        orientation = simd_mul(r, orientation);
    }

    // Produce a view matrix
    simd::float4x4 getViewMatrix() {
        // The third column of orientation is "negative forward",
        // but let's recalc carefully for clarity
        simd::float3 forward = getForwardVector();
        simd::float3 up      = getUpVector();
        simd::float3 center  = position + forward;
        //printf("<%f, %f, %f>\n", forward.x, forward.y, forward.z);

        return lookAt(position, center, up);
    }

private:
    simd::float4x4 makeIdentity4x4() {
        return simd::float4x4{
            simd::float4{ 1, 0, 0, 0},
            simd::float4{ 0, 1, 0, 0},
            simd::float4{ 0, 0, 1, 0},
            simd::float4{ 0, 0, 0, 1}
        };
    }
    
    // Build a rotation matrix that rotates by 'angle' around 'axis'
    simd::float4x4 makeRotation(simd::float3 axis, float angle) {
        axis = simd::normalize(axis);
        float c = cosf(angle);
        float s = sinf(angle);
        float mc = 1.0f - c;

        float x = axis.x;
        float y = axis.y;
        float z = axis.z;

        // Rodrigues' rotation formula in matrix form
        return simd::float4x4{
            simd::float4{ c + x*x*mc,   x*y*mc + z*s, x*z*mc - y*s, 0},
            simd::float4{ x*y*mc - z*s, c + y*y*mc,   y*z*mc + x*s, 0},
            simd::float4{ x*z*mc + y*s, y*z*mc - x*s, c + z*z*mc,   0},
            simd::float4{ 0,           0,           0,            1}
        };
    }

    // Return local axes from orientation
    // orientation is a column-major matrix:
    //   col0 = local X
    //   col1 = local Y
    //   col2 = local Z
    //   col3 = translation (unused here)
    simd::float3 getRightVector()   { return simd::normalize(orientation.columns[0].xyz); }
    simd::float3 getUpVector()      { return simd::normalize(orientation.columns[1].xyz); }
    simd::float3 getForwardVector() {
        // Typically forward is -Z in our row, so we might use negative col2
        // or define forward as col2, depending on your convention
        return simd::normalize(-orientation.columns[2].xyz);
    }

    // Standard lookAt
    simd::float4x4 lookAt(simd::float3 eye, simd::float3 center, simd::float3 up) {
        simd::float3 f = simd::normalize(center - eye);
        simd::float3 r = simd::normalize(simd::cross(up, f));
        simd::float3 u = simd::cross(f, r);

        simd::float4x4 m = {
            simd::float4{r.x, u.x, f.x, 0},
            simd::float4{r.y, u.y, f.y, 0},
            simd::float4{r.z, u.z, f.z, 0},
            simd::float4{-simd::dot(r, eye),
                         -simd::dot(u, eye),
                         -simd::dot(f, eye), 1}
        };
        return m;
    }
};
