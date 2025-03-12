#include <metal_stdlib>

#include "common.metal"

using namespace metal;

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


float activate(const float x, const uint act);
float activate_derivative(const float y, const uint act);



/**
 * forward_batch_norm
 *
 * This kernel computes the forward pass of Batch Normalization.
 * It calculates per-batch mean/variance for each feature if isTraining = true,
 * and uses them to normalize. It also updates “runningMean”/“runningVariance”
 * and *stores the exact batch stats in “savedMean”/“savedVariance”* for the backward pass.
 *
 * Buffers (in order):
 *  0) input:
 *     The activations/input to this BN layer, shape [batchSize * featureDim].
 *  1) output:
 *     The normalized output after BN, same shape as input.
 *  2) gamma:
 *     The per-feature scale vector, length = featureDim.
 *  3) beta:
 *     The per-feature shift vector, length = featureDim.
 *  4) runningMean:
 *     The running (moving) mean, length = featureDim.
 *     Updated each time if isTraining = true.
 *  5) runningVariance:
 *     The running variance, length = featureDim.
 *     Updated each time if isTraining = true.
 *  6) savedMean:
 *     A buffer (length = featureDim) to store the *exact batch mean* for the backward pass.
 *  7) savedVariance:
 *     A buffer (length = featureDim) to store the *exact batch variance* for the backward pass.
 *  8) epsilon:
 *     A float for numeric stability in the denominator (e.g. 1e-5).
 *  9) featureDim:
 *     The number of features (channels).
 * 10) isTraining:
 *     Boolean: true => training mode (use batch stats), false => inference (use running stats).
 * 11) batchSize:
 *     The number of samples in the batch.
 * 12) debug:
 *     A debug buffer (float*) if needed. Not used here.
 *
 * Thread info:
 *  - thread_position_in_grid (gid) => which feature index [0..featureDim-1]
 */
kernel void forward_batch_norm(
    device const float*  input           [[buffer(0)]],
    device float*        output          [[buffer(1)]],
    device const float*  gamma           [[buffer(2)]],
    device const float*  beta            [[buffer(3)]],
    device float*        runningMean     [[buffer(4)]],
    device float*        runningVariance [[buffer(5)]],
    device float*        savedMean       [[buffer(6)]],
    device float*        savedVariance   [[buffer(7)]],
    constant float&      epsilon         [[buffer(8)]],
    constant int&        featureDim      [[buffer(9)]],
    constant bool&       isTraining      [[buffer(10)]],
    constant uint&       batchSize       [[buffer(11)]],
    device float*        debug           [[buffer(12)]],  // not used
    uint                 gid             [[thread_position_in_grid]]
)
{
    if ((int)gid >= featureDim) return;

    // 1) Compute per-batch mean for this feature
    float sum = 0.0f;
    for (uint b = 0; b < batchSize; b++) {
        sum += input[b * featureDim + gid];
    }
    float batchMean = sum / float(batchSize);

    // 2) Compute per-batch variance
    float sqSum = 0.0f;
    for (uint b = 0; b < batchSize; b++) {
        float diff = input[b * featureDim + gid] - batchMean;
        sqSum += diff * diff;
    }
    float batchVar = sqSum / float(batchSize);

    // 3) Save these batch stats for backward pass, no matter what
    savedMean[gid]     = batchMean;
    savedVariance[gid] = batchVar;

    // 4) Update running stats if training
    if (isTraining) {
        // Typically: running = momentum*running + (1 - momentum)*new
        float momentum = 0.9f;
        runningMean[gid]     = momentum * runningMean[gid]     + (1.0f - momentum) * batchMean;
        runningVariance[gid] = momentum * runningVariance[gid] + (1.0f - momentum) * batchVar;
    }

    // 5) Decide which mean and variance to use
    float usedMean     = isTraining ? batchMean  : runningMean[gid];
    float usedVariance = isTraining ? batchVar   : runningVariance[gid];
    float invStd       = rsqrt(usedVariance + epsilon);

    // 6) Normalize each sample’s value
    for (uint b = 0; b < batchSize; b++) {
        float val  = input[b * featureDim + gid];
        float norm = (val - usedMean) * invStd;
        output[b * featureDim + gid] = gamma[gid] * norm + beta[gid];
    }
}


/**
 * backward_batch_norm
 *
 * This kernel computes the backward pass for Batch Normalization.
 * It uses the *saved* batch mean/variance from the forward pass if training,
 * or the running stats if not. Then it computes the gradients w.r.t. gamma/beta
 * and updates them. Finally, it computes the “output errors” (a.k.a. ∂L/∂(input))
 * for this BN layer.
 *
 * We keep your “inputErrors” = the errors fed into this layer by the next layer,
 * and “outputErrors” = the errors fed back from this layer to the previous layer.
 *
 * Buffers (in the order of the buffer indices):
 *  0) input:
 *     The original forward-pass input data [batchSize * featureDim].
 *  1) inputErrors:
 *     The gradient from the next layer: ∂L/∂(BN output), shape [batchSize * featureDim].
 *  2) outputErrors:
 *     The gradient we produce to feed back to the previous layer: ∂L/∂(BN input),
 *     shape [batchSize * featureDim].
 *  3) gamma:
 *     The scale vector, length = featureDim. We will update gamma in place.
 *  4) beta:
 *     The shift vector, length = featureDim. We will update beta in place.
 *  5) savedMean:
 *     The exact batch mean from the forward pass, length = featureDim.
 *     (Used if isTraining = true).
 *  6) savedVariance:
 *     The exact batch variance from the forward pass, length = featureDim.
 *     (Used if isTraining = true).
 *  7) runningMean:
 *     The running mean array, length = featureDim.
 *     (Used if isTraining = false).
 *  8) runningVariance:
 *     The running variance array, length = featureDim.
 *     (Used if isTraining = false).
 *  9) epsilon:
 *     Float for numerical stability (e.g. 1e-5).
 * 10) featureDim:
 *     The number of features.
 * 11) isTraining:
 *     Boolean: true => use saved batch stats; false => use running stats.
 * 12) batchSize:
 *     Number of samples in the batch.
 * 13) learningRate:
 *     Float for updating gamma/beta.
 * 14) debug:
 *     A debug float buffer if needed (not used here).
 *
 * Typically, we dispatch with gridSize >= featureDim so each thread handles one feature index.
 */
kernel void backward_batch_norm(
    device const float*  input            [[buffer(0)]],
    device const float*  inputErrors      [[buffer(1)]],
    device float*        outputErrors     [[buffer(2)]],
    device float*        gamma            [[buffer(3)]],
    device float*        beta             [[buffer(4)]],
    device const float*  savedMean        [[buffer(5)]],
    device const float*  savedVariance    [[buffer(6)]],
    device const float*  runningMean      [[buffer(7)]],
    device const float*  runningVariance  [[buffer(8)]],
    constant float&      epsilon          [[buffer(9)]],
    constant int&        featureDim       [[buffer(10)]],
    constant bool&       isTraining       [[buffer(11)]],
    constant uint&       batchSize        [[buffer(12)]],
    constant float&      learningRate     [[buffer(13)]],
    device float*        debug            [[buffer(14)]],
    uint                 gid              [[thread_position_in_grid]]
)
{
    if ((int)gid >= featureDim) return;

    // 1) Decide which mean/variance to use:
    float mean     = isTraining ? savedMean[gid]     : runningMean[gid];
    float var      = isTraining ? savedVariance[gid] : runningVariance[gid];
    float invStd   = rsqrt(var + epsilon);

    // 2) Accumulate sum(dY) and sum(dY*xhat) for this feature
    float sum_dY      = 0.0f;
    float sum_dY_xhat = 0.0f;

    for (uint b = 0; b < batchSize; b++) {
        float x    = input[b * featureDim + gid];
        float xhat = (x - mean) * invStd;
        float dY   = inputErrors[b * featureDim + gid];
        sum_dY      += dY;
        sum_dY_xhat += (dY * xhat);
    }

    // 3) Gradients for gamma/beta
    float grad_gamma = sum_dY_xhat;
    float grad_beta  = sum_dY;

    grad_gamma /= float(batchSize);
    grad_beta  /= float(batchSize);

    // Update gamma, beta in place
    gamma[gid] -= learningRate * grad_gamma;
    beta[gid]  -= learningRate * grad_beta;

    // 4) Now compute outputErrors: ∂L/∂(BN input)
    //    Formula:
    //     dX = (1/N) * gamma * invStd * [ N*dY - sum(dY) - xhat * sum(dY*xhat) ]
    for (uint b = 0; b < batchSize; b++) {
        float x    = input[b * featureDim + gid];
        float xhat = (x - mean) * invStd;
        float dY   = inputErrors[b * featureDim + gid];

        float dX = (gamma[gid] * invStd / float(batchSize)) *
                   (float(batchSize) * dY - sum_dY - xhat * sum_dY_xhat);

        outputErrors[b * featureDim + gid] = dX;
    }
}
