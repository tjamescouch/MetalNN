#ifndef MULTI_LAYER_KERNELS_H
#define MULTI_LAYER_KERNELS_H

#pragma region Declarations {

namespace multilayerkernels {

const inline char* nnKernelSrc = R"(
#include <metal_stdlib>
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

// Clamping thresholds
constant float max_abs_sum = 1000.0f;
constant float threshold    = 1.0f;
constant float decay_factor = 0.9999f;

/**
 * A utility function to apply an activation.
 *
 * x: the input activation value
 * act: which activation to apply (e.g. ACTIVATION_RELU)
 */
inline float activate(const float x, const uint act) {
    switch (act) {
        case ACTIVATION_LINEAR:  return x;
        case ACTIVATION_RELU:    return max(0.0f, x);
        case ACTIVATION_TANH:    return tanh(x);
        case ACTIVATION_SIGMOID: return 1.0f / (1.0f + exp(-x));
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

/**
 * forward_dense_layer
 *
 * This kernel computes the forward pass of a dense (fully-connected) layer.
 * 
 * Buffers (in order):
 *  0) h:    The input activations to this layer (shape: [batchSize * hidden_dim]).
 *           Index as h[sample_id * hidden_dim + i].
 *  1) y:    The output activations of this layer (shape: [batchSize * output_dim]).
 *           We'll fill y with activated outputs.
 *  2) W:    The weight matrix (shape: [hidden_dim * output_dim]), row-major:
 *           W[i * output_dim + neuron_id].
 *  3) b:    The bias vector (shape: [output_dim]).
 *  4) pH:   A single uint specifying the hidden_dim (the # of inputs per neuron).
 *  5) pN:   A single uint specifying the output_dim (the # of neurons in this layer).
 *  6) activation: A single uint specifying the activation type (RELU, TANH, etc.).
 *  7) batchSize: The # of samples in the batch (uint).
 *  8) debug: A debug float buffer [batchSize * output_dim], for optional logging if desired.
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 *
 * We typically dispatch with gridSize = batchSize * output_dim,
 * so each thread handles (sample_id, neuron_id).
 */
kernel void forward_dense_layer(
    device const float* h         [[buffer(0)]],  // Input activations
    device       float* y         [[buffer(1)]],  // Output activations
    device const float* W         [[buffer(2)]],  // Weights
    device const float* b         [[buffer(3)]],  // Biases
    device const uint* pH         [[buffer(4)]],  // hidden_dim
    device const uint* pN         [[buffer(5)]],  // output_dim
    device const uint* activation [[buffer(6)]],  // Activation type
    constant uint& batchSize      [[buffer(7)]],  // # of samples
    device float* debug           [[buffer(8)]],  // Debug buffer
    uint tid                      [[thread_position_in_threadgroup]],
    uint gid                      [[thread_position_in_grid]]
)
{
    uint hidden_dim = *pH;
    uint output_dim = *pN;
    uint act_type   = *activation;

    // sample_id, neuron_id
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batchSize || neuron_id >= output_dim) return;

    // We'll use a threadgroup array for partial storage
    threadgroup float shared_y[1024];

    // 1) Compute the pre-activation sum
    float sum = b[neuron_id];
    for (uint i = 0; i < hidden_dim; ++i) {
        float inputVal = h[sample_id * hidden_dim + i];
        float weightVal = W[i * output_dim + neuron_id];
        sum += inputVal * weightVal;
    }

    // Clamp to avoid numerical blow-up
    sum = clamp(sum, -max_abs_sum, max_abs_sum);

    // Store in threadgroup memory to do optional softmax
    shared_y[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) If activation is softmax, handle it per sample
    if (act_type == ACTIVATION_SOFTMAX) {
        // We do a per-sample softmax in threadgroup memory
        uint sample_offset = sample_id * output_dim;
        // Find max for numeric stability
        float maxVal = shared_y[sample_offset];  // We assume sample_offset+0 is in range
        for (uint i = 1; i < output_dim; ++i) {
            float val = shared_y[sample_offset + i];
            maxVal = max(maxVal, val);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exponentiate each
        shared_y[tid] = exp(shared_y[tid] - maxVal);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One thread (neuron_id=0) sums and normalizes
        if (neuron_id == 0) {
            float sumExp = 0.0f;
            for (uint i = 0; i < output_dim; ++i) {
                sumExp += shared_y[sample_offset + i];
            }
            // Normalize
            for (uint i = 0; i < output_dim; ++i) {
                shared_y[sample_offset + i] /= sumExp;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Optionally store debug
        debug[gid] = shared_y[tid];
    }
    else {
        // For non-softmax, apply the chosen activation in place
        shared_y[tid] = activate(shared_y[tid], act_type);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Write the result to the output array
    y[gid] = shared_y[tid];
}


/**
 * learn_non_terminal_dense_layer
 *
 * This kernel handles the backward pass for a dense layer that is NOT the final
 * layer (i.e., it receives "input errors" from the next layer). 
 *
 * Buffers (in order):
 *  0) h:             The input activations fed into this dense layer [batchSize * input_dim].
 *  1) W:             The weight matrix for this layer [input_dim * output_dim].
 *  2) b:             The bias vector [output_dim].
 *  3) y_hat:         The output activations from this layer [batchSize * output_dim].
 *  4) inputErrors:   The "errors" fed INTO the current layer from the next layer [batchSize * output_dim].
 *                    (Sometimes called dY or "output gradient of next layer.")
 *  5) outputError:   The "errors" fed BACK by this layer to the previous layer [batchSize * output_dim].
 *                    However, here we only store each neuron's partial ∂L/∂(neuron output).
 *  6) input_dim:     # of inputs per neuron (uint).
 *  7) output_dim:    # of neurons in this layer (uint).
 *  8) pDecay:        A float pointer for weight decay or momentum factor (decay).
 *  9) activation:    The activation type for this layer (uint).
 * 10) debug:         Debug buffer [batchSize * output_dim].
 * 11) prevLayerErrors: The "output errors" from this layer to the PREVIOUS layer [batchSize * input_dim].
 *                      (i.e., ∂L/∂(h), shape [batchSize * input_dim]).
 * 12) batch_size:    The number of samples in the batch (uint).
 * 13) pLearningRate: A float pointer giving the learning rate.
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 *
 * Typically, dispatch with gridSize = batchSize * output_dim.
 * Each thread corresponds to (sample_id, neuron_id).
 */
kernel void learn_non_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // input activations
    device float*       W                [[buffer(1)]],  // weights
    device const float* b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // layer outputs
    device const float* inputErrors      [[buffer(4)]],  // errors fed INTO this layer from next
    device float*       outputError      [[buffer(5)]],  // errors fed BACK to previous (delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    device float*       pDecay           [[buffer(8)]],  // e.g., momentum or weight decay factor
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // final error to the previous layer's activations
    constant uint&      batch_size       [[buffer(12)]],
    device const float* pLearningRate    [[buffer(13)]], 
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay         = *pDecay;
    float learning_rate = *pLearningRate;

    // Pointers to this sample's portion of each buffer
    const device float* sample_h         = h + (sample_id * input_dim);
    const device float* sample_y_hat     = y_hat + (sample_id * output_dim);
    const device float* sample_inErrors  = inputErrors + (sample_id * output_dim);
    device float*       sample_outError  = outputError + (sample_id * output_dim);
    device float*       sample_prevError = prevLayerErrors + (sample_id * input_dim);

    // raw_error is the incoming gradient wrt this neuron's output
    float raw_error = sample_inErrors[neuron_id];

    // Delta = error * activation derivative
    float dAct = activate_derivative(sample_y_hat[neuron_id], activation);
    float delta = raw_error * dAct;

    // Clamp to avoid numerical blow-up
    delta = clamp(delta, -threshold, threshold);

    // Save partial delta in outputError (i.e., error from this layer's viewpoint)
    sample_outError[neuron_id] = delta;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For each input in [0..input_dim-1], update weight and accumulate error
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;  // partial dW

        // clamp grad
        //grad = clamp(grad, -threshold, threshold);


        debug[i * output_dim + neuron_id] = grad;

        // We want to read the old weight, THEN do an atomic add
        device atomic_float* wAddr = (device atomic_float*)&W[i * output_dim + neuron_id];

        float oldWeight = atomic_load_explicit(wAddr, memory_order_relaxed);

        // Atomic update the weight
        // W -= learning_rate * grad * decay
        float wUpdate = -learning_rate * grad * decay;
        atomic_fetch_add_explicit(wAddr, wUpdate, memory_order_relaxed);

        // For the error to pass back to previous layer: oldWeight * delta
        // Use oldWeight from before we updated it.
        float prevErrTerm = oldWeight * delta;
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevError[i],
                                  prevErrTerm, memory_order_relaxed);
    }

    // Update bias
    device atomic_float* bAddr = (device atomic_float*)&b[neuron_id];
    float biasUpdate = -learning_rate * delta * decay;
    atomic_fetch_add_explicit(bAddr, biasUpdate, memory_order_relaxed);
}


/**
 * learn_terminal_dense_layer
 *
 * This kernel handles the backward pass for a dense layer that IS the final layer.
 * Instead of "inputErrors," it receives the direct target difference (y_hat - y)
 * or some derivative thereof. We call that "inputErrors" in a final layer, but
 * you have it as the "targets" array. We'll keep to your naming scheme:
 * 
 * Buffers (in order):
 *  0) h:      input activations for this final layer [batchSize * input_dim].
 *  1) W:      the weight matrix [input_dim * output_dim].
 *  2) b:      the bias vector [output_dim].
 *  3) y_hat:  the predicted outputs for this layer [batchSize * output_dim].
 *  4) y:      the ground-truth targets [batchSize * output_dim].
 *  5) error:  the partial errors from this final layer's perspective [batchSize * output_dim].
 *  6) input_dim:  # of inputs per neuron (uint).
 *  7) output_dim: # of neurons in this layer (uint).
 *  8) pDecay:     float pointer for weight decay or momentum factor.
 *  9) activation: the activation type for this final layer (uint).
 * 10) debug:      debug buffer [batchSize * output_dim].
 * 11) prevLayerErrors: the "output errors" from this final layer to the previous layer [batchSize * input_dim].
 * 12) batch_size: how many samples in the batch (uint).
 * 13) pLearningRate: the learning rate (float pointer).
 *
 * Thread positions:
 *  tid: thread_position_in_threadgroup
 *  gid: thread_position_in_grid
 */
kernel void learn_terminal_dense_layer(
    device const float* h                [[buffer(0)]],  // final layer input activations
    device float*       W                [[buffer(1)]],  // weights
    device float*       b                [[buffer(2)]],  // biases
    device const float* y_hat            [[buffer(3)]],  // predicted outputs
    device const float* y                [[buffer(4)]],  // targets
    device float*       error            [[buffer(5)]],  // partial error (this final layer's delta)
    constant uint&      input_dim        [[buffer(6)]],
    constant uint&      output_dim       [[buffer(7)]],
    device float*       pDecay           [[buffer(8)]],
    constant uint&      activation       [[buffer(9)]],
    device float*       debug            [[buffer(10)]],
    device float*       prevLayerErrors  [[buffer(11)]], // error to previous layer
    constant uint&      batch_size       [[buffer(12)]],
    device const float* pLearningRate    [[buffer(13)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[thread_position_in_grid]]
)
{
    // Identify sample + neuron
    uint sample_id = gid / output_dim;
    uint neuron_id = gid % output_dim;

    if (sample_id >= batch_size || neuron_id >= output_dim) return;

    float decay         = *pDecay;
    float learning_rate = *pLearningRate;

    // Pointers to this sample's data
    const device float* sample_h     = h     + (sample_id * input_dim);
    const device float* sample_y_hat = y_hat + (sample_id * output_dim);
    const device float* sample_y     = y     + (sample_id * output_dim);
    device float*       sample_error = error + (sample_id * output_dim);
    device float*       sample_prevE = prevLayerErrors + (sample_id * input_dim);

    // For terminal layer, raw_error is typically (y_hat - y).
    // We'll clamp, then multiply by derivative if not softmax.
    float raw_error = sample_y_hat[neuron_id] - sample_y[neuron_id];
    debug[gid] = raw_error;

    float delta;
    if (activation == ACTIVATION_SOFTMAX) {
        // In typical frameworks, for cross-entropy + softmax => delta = (y_hat - y).
        delta = raw_error;
    } else {
        // Multiply by derivative of the activation
        float dAct = activate_derivative(sample_y_hat[neuron_id], activation);
        delta = raw_error * dAct;
    }

    // clamp
    delta = clamp(delta, -threshold, threshold);
    // Save in error buffer
    sample_error[neuron_id] = delta;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weight + bias update
    for (uint i = 0; i < input_dim; i++) {
        float grad = sample_h[i] * delta;
        grad = clamp(grad, -threshold, threshold);

        device atomic_float* wAddr = (device atomic_float*)&W[i * output_dim + neuron_id];
        float oldWeight = atomic_load_explicit(wAddr, memory_order_relaxed);

        float wUpdate = -learning_rate * grad * decay;
        atomic_fetch_add_explicit(wAddr, wUpdate, memory_order_relaxed);

        // error to feed back
        float prevErrTerm = oldWeight * delta;
        atomic_fetch_add_explicit((device atomic_float*)&sample_prevE[i], prevErrTerm, memory_order_relaxed);
    }

    // Bias
    device atomic_float* bAddr = (device atomic_float*)&b[neuron_id];
    float biasUpdate = -learning_rate * delta * decay;
    atomic_fetch_add_explicit(bAddr, biasUpdate, memory_order_relaxed);
}

//-------------------------------------------------------------------
// Forward pass for the recurrent layer (RNN cell)
kernel void forward_rnn(
    device const float* x            [[buffer(0)]],
    device       float* h_prev       [[buffer(1)]],
    device       float* h            [[buffer(2)]],
    device const float* W_xh         [[buffer(3)]],
    device const float* W_hh         [[buffer(4)]],
    device const float* b            [[buffer(5)]],
    device const uint* pX            [[buffer(6)]],
    device const uint* pH            [[buffer(7)]],
    device const uint* activation    [[buffer(8)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;

    if (tid >= hidden_dim) return;
    
    float sum = b[tid];
    
    // Contribution from current input
    for (uint i = 0; i < input_dim; i++) {
        sum += x[i] * W_xh[i * hidden_dim + tid];
    }
    
    // Recurrent contribution from previous hidden state
    for (uint j = 0; j < hidden_dim; j++) {
        sum += h_prev[j] * W_hh[j * hidden_dim + tid];
    }
    
    h_prev[tid] = h[tid];
    h[tid] = activate(sum, *activation);
}

//-------------------------------------------------------------------
// Learning kernel for the recurrent layer (multi-step BPTT)
//-------------------------------------------------------------------
kernel void learn_rnn(
    device const float* x            [[buffer(0)]],
    device const float* h_prev       [[buffer(1)]],
    device       float* W_xh         [[buffer(2)]],
    device       float* W_hh         [[buffer(3)]],
    device       float* b            [[buffer(4)]],
    device const float* h            [[buffer(5)]],
    device       float* output_error [[buffer(6)]],
    device const float* next_hidden_error [[buffer(7)]], 
    device       float* hidden_error [[buffer(8)]],
    device const uint* pX            [[buffer(9)]],
    device const uint* pH            [[buffer(10)]],
    device       float* pDecay       [[buffer(11)]],
    device const uint* activation    [[buffer(12)]],
    constant     uint& batch_size    [[buffer(13)]],
    constant     float& learning_rate [[buffer(14)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint input_dim = *pX;
    uint hidden_dim = *pH;

    if (tid >= hidden_dim) return;

    if (tid == 0) {
        *pDecay *= decay_factor;
    }
    float decay = *pDecay;


    // Combine the next timestep's hidden error plus local output_error
    float accumulated_err = output_error[tid];
    for (uint k = 0; k < hidden_dim; k++) {
        accumulated_err += next_hidden_error[k] * W_hh[k * hidden_dim + tid];
    }

    // Multiply by activation derivative of current hidden state
    float delta = accumulated_err * activate_derivative(h[tid], *activation);
    delta = clamp(delta, -threshold, threshold);
    hidden_error[tid] = delta;
    output_error[tid] = delta;

    // Update input-to-hidden weights
    for (uint i = 0; i < input_dim; i++) {
        W_xh[i * hidden_dim + tid] -= learning_rate * delta * x[i] * decay;
    }

    // Update recurrent weights
    for (uint j = 0; j < hidden_dim; j++) {
        W_hh[j * hidden_dim + tid] -= learning_rate * delta * h_prev[j] * decay;
    }

    // Update bias
    b[tid] -= learning_rate * delta * decay;
}

 kernel void forward_dropout(
     device const float* input       [[buffer(0)]],
     device       float* output      [[buffer(1)]],
     device const float* randomMask  [[buffer(2)]], // binary mask: 0 or 1
     constant float& rate        [[buffer(3)]],
     constant int&   featureDim  [[buffer(4)]],
     constant bool&      isTraining  [[buffer(5)]],
     device float*       debug       [[buffer(6)]],
     uint                tid         [[thread_position_in_grid]]
 ) {
     if ((int)tid >= featureDim) return;
 
     float x        = input[tid];
     float maskVal  = randomMask[tid]; // binary mask value

    if (tid < 100) {
        debug[tid] = input[tid];  // BEFORE dropout scaling
    }
 
     if (isTraining) {
         output[tid] = (maskVal > 0.5f) ? (x / (1.0f - rate)) : 0.0f;
     } else {
         output[tid] = x;
     }

 }

//-------------------------------------------------------------------
// Backward pass for Dropout layer
 kernel void backward_dropout(
     device const float* input_error  [[buffer(0)]],
     device float*       output_error  [[buffer(1)]],
     device const float* randomMask   [[buffer(2)]],
     constant float& rate         [[buffer(3)]],
     constant uint&  featureDim   [[buffer(4)]],
     device float* debug   [[buffer(5)]],
     uint tid                         [[thread_position_in_grid]]
 ) {
     if ((int)tid >= featureDim) return;
 
     float maskVal  = randomMask[tid]; // binary mask value
     output_error[tid] = (maskVal > 0.5) ? (input_error[tid] / (1.0f - rate)) : 0.0f;
 }


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

kernel void adam_kernel(
    device float* parameters         [[buffer(0)]],   // Parameters (weights or biases)
    device float* gradients          [[buffer(1)]],   // Gradients accumulated across the batch
    device float* m                  [[buffer(2)]],   // First moment vector
    device float* v                  [[buffer(3)]],   // Second moment vector
    constant float& learning_rate    [[buffer(4)]],   // Base learning rate
    constant float& beta1            [[buffer(5)]],   // Exponential decay rate for first moment
    constant float& beta2            [[buffer(6)]],   // Exponential decay rate for second moment
    constant float& epsilon          [[buffer(7)]],   // Prevent division by zero
    constant uint& batch_size        [[buffer(8)]],   // Current batch size
    constant uint& timestep          [[buffer(9)]],   // Global step for bias correction
    constant uint& param_count       [[buffer(10)]],  // Total number of parameters
    uint tid                         [[thread_position_in_grid]]
)
{
    // Each thread handles one parameter index
    if (tid >= param_count) return;

    // 1) Average the gradient over the batch (assuming 'gradients' is the sum)
    float grad_avg = gradients[tid] / (float)batch_size;

    // 2) Update biased first moment estimate (m) and second moment estimate (v)
    m[tid] = beta1 * m[tid] + (1.0f - beta1) * grad_avg;
    v[tid] = beta2 * v[tid] + (1.0f - beta2) * grad_avg * grad_avg;

    // 3) Skip explicit bias correction; keep old m & v directly.
    float m_hat = m[tid];
    float v_hat = v[tid];

    // 4) Compute the Adam update (no bias correction)
    float update = learning_rate * (m_hat / (sqrt(v_hat) + epsilon));

    // Optional clamp to prevent extreme updates
    update = clamp(update, -1e3f, 1e3f);

    // 5) Apply the update
    parameters[tid] -= update;

    // 6) Reset gradient accumulator for next batch
    gradients[tid] = 0.0f;
}

// Constants defining reduction types for clarity
kernel void forward_map_reduce(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant int& input_size         [[buffer(2)]],
    constant uint& reductionType     [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    threadgroup float sharedData[1024];

    // Load data into threadgroup memory
    sharedData[tid] = ((int)tid < input_size) ? input[tid] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction operation
    switch (reductionType) {
        case REDUCTION_SUM:
        case REDUCTION_MEAN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] += sharedData[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = (reductionType == REDUCTION_MEAN) ? (sharedData[0] / input_size) : sharedData[0];
            }
            break;

        case REDUCTION_MAX:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_MIN:
            for (uint stride = input_size / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                output[0] = sharedData[0];
            }
            break;

        case REDUCTION_SOFTMAX:
            if (tid == 0) {
                float maxVal = input[0];
                for (int i = 1; i < input_size; ++i) {
                    maxVal = max(maxVal, input[i]);
                }

                float sumExp = 0.0f;
                for (int i = 0; i < input_size; ++i) {
                    sumExp += exp(input[i] - maxVal);
                }

                for (int i = 0; i < input_size; ++i) {
                    output[i] = exp(input[i] - maxVal) / sumExp;
                }
            }
            break;
    }
}

kernel void backward_map_reduce(
    device const float* outputDelta  [[buffer(0)]],
    device const float* forwardOutput[[buffer(1)]],
    device float* inputErrors        [[buffer(2)]],
    constant uint& input_size        [[buffer(3)]],
    constant uint& reductionType     [[buffer(4)]],
    uint tid                         [[thread_position_in_grid]]
) {
    if (tid >= input_size) return;

    switch (reductionType) {
        case REDUCTION_SUM:
            inputErrors[tid] = outputDelta[0];
            break;

        case REDUCTION_MEAN:
            inputErrors[tid] = outputDelta[0] / input_size;
            break;

        case REDUCTION_MAX:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_MIN:
            inputErrors[tid] = (forwardOutput[tid] == outputDelta[1]) ? outputDelta[0] : 0.0f;
            break;

        case REDUCTION_SOFTMAX:
            // forwardOutput is already softmaxed values here
            float grad = 0.0f;
            for (uint i = 0; i < input_size; ++i) {
                float indicator = (tid == i) ? 1.0f : 0.0f;
                grad += outputDelta[i] * forwardOutput[tid] * (indicator - forwardOutput[i]);
            }
            inputErrors[tid] = grad;
            break;
    }
}

)";

} // namespace multilayerkernels

#pragma endregion Declarations }
#endif
