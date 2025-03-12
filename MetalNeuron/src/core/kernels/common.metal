#ifndef common_metal
#define common_metal

#include <metal_stdlib>

#define REDUCTION_SUM     0
#define REDUCTION_MEAN    1
#define REDUCTION_MAX     2
#define REDUCTION_MIN     3
#define REDUCTION_SOFTMAX 4


#define ACTIVATION_LINEAR  0
#define ACTIVATION_RELU    1
#define ACTIVATION_TANH    2
#define ACTIVATION_SIGMOID 3
#define ACTIVATION_SOFTMAX 4



/**
 * A utility function to apply an activation.
 *
 * x: the input activation value
 * act: which activation to apply (e.g. ACTIVATION_RELU)
 */
inline float activate(const float x, const uint act) {
    switch (act) {
        case ACTIVATION_LINEAR:  return x;
        case ACTIVATION_RELU:    return metal::max(0.0f, x);
        case ACTIVATION_TANH:    return metal::tanh(x);
        case ACTIVATION_SIGMOID: return 1.0f / (1.0f + metal::exp(-x));
        default:                 return 0.0f;  // Fallback
    }
}

/**
 * Derivative of the activation function.
 *
 * y: the already-activated value (e.g. y = activate(x, act))
 * act: which activation (e.g. ACTIVATION_RELU)
 */
inline float activate_derivative(const float y, const uint act) {
    switch (act) {
        case ACTIVATION_LINEAR:  return 1.0f;
        case ACTIVATION_RELU:    return (y > 0.0f) ? 1.0f : 0.0f;
        case ACTIVATION_TANH:    return 1.0f - (y * y);
        case ACTIVATION_SIGMOID: return y * (1.0f - y);
        default:                 return 0.0f;
    }
}

#endif
