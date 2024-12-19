#define FLT_MIN 1.175494e-38
#include <cuda_bf16.h>

// The compiler does not properly optimize this unless we use the CPP to make
// the constant explicit.
#define warpSize 32

extern "C" __global__ void fp32_to_bf16(const float* input, __nv_bfloat16* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = __float2bfloat16_rz(input[i]);
    }
}

extern "C" __global__ void batched_softmax(float *data, int size, int batch_size) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize;

    int batch_id = blockIdx.x; // Simplified batch ID, assumes one block per batch

    if (batch_id >= batch_size) return;

    float* x = data + batch_id * size;

    // ----- 1. Partial Max Value Calculation -----
    float partial_max = FLT_MIN; // Initialize to minimum float
    for (int i = tid; i < size; i += blockDim.x) {
        partial_max = fmaxf(x[i], partial_max);
    }

    // ----- 2. Warp Reduction for Max -----
    float warpMax = partial_max;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float otherValue = __shfl_down_sync(0xFFFFFFFF, warpMax, offset);
        warpMax = fmaxf(warpMax, otherValue);
    }

    // ----- 3. Warp Leader Writes to Shared Memory -----
    if (laneId == 0) {
        shared[warpId] = warpMax;
    }
    __syncthreads();

    // ----- 4. Block Reduction for Max -----
    float blockMax = shared[laneId];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      float otherValue = __shfl_down_sync(0xFFFFFFFF, blockMax, offset);
      blockMax = fmaxf(blockMax, otherValue);
    }

    // ----- 5. Broadcast the block max -----
    float maxVal = __shfl_sync(0xFFFFFFFF, blockMax, 0);

    // ----- 6. Calculate Partial Exp and Sum -----
    float partial_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] = __expf(x[i] - maxVal);
        partial_sum += x[i];
    }

    // ----- 7. Warp Reduction for Sum -----
    float warpSum = partial_sum;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float otherValue = __shfl_down_sync(0xFFFFFFFF, warpSum, offset);
        warpSum += otherValue;
    }

    // ----- 8. Warp Leader Writes Sum to Shared Memory -----
    if (laneId == 0) {
        shared[warpId] = warpSum;
    }
    __syncthreads();

    // ----- 9. Block Reduction for Sum -----
    float blockSum = shared[laneId];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      float otherValue = __shfl_down_sync(0xFFFFFFFF, blockSum, offset);
      blockSum += otherValue;
    }

    // ----- 10. Broadcast the block Sum -----
    float sum = __shfl_sync(0xFFFFFFFF, blockSum, 0);

    // ----- 11. Normalize each element -----
    float rsum = 1.0f / sum;
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] *= rsum;
    }
}

extern "C" __global__ void swiGLU(float* hb, float* hb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = hb[i];
        // silu(x) = x * sigmoid(x)
        val *= 1.0f / (1.0f + __expf(-val));  //More efficient sigmoid calculation
        val *= hb2[i];
        hb[i] = val;
    }
}

extern "C" __global__ void rope_rotary_encoding(
    float* __restrict__ q,        // Query matrix
    float* __restrict__ k,        // Key matrix
    const int n_heads,            // Number of attention heads
    const int n_kv_heads,         // Number of key-value heads
    const int head_size,          // Size of each head (assumed to be even)
    const int pos)                // Position index
{
    // Each thread computes a specific (head, j) pair
    int head = blockIdx.x;          // Head index (i)
    int j = threadIdx.x * 2;        // Position within head (stride of 2 for complex pairs)

    if (head >= n_heads || j >= head_size)
        return;

    // Compute the frequency for this position
    float freq = __powf(500000.0f, (float)-j / (float)head_size);
    float val = pos * freq;
    float fcr = __cosf(val);
    float fci = __sinf(val);

    // Load q values (q0 and q1)
    float q0 = q[head * head_size + j];
    float q1 = q[head * head_size + j + 1];

    // Rotate q values
    q[head * head_size + j]     = q0 * fcr - q1 * fci;
    q[head * head_size + j + 1] = q0 * fci + q1 * fcr;

    // Rotate k values only if within n_kv_heads
    if (head < n_kv_heads) {
        float k0 = k[head * head_size + j];
        float k1 = k[head * head_size + j + 1];

        k[head * head_size + j]     = k0 * fcr - k1 * fci;
        k[head * head_size + j + 1] = k0 * fci + k1 * fcr;
    }
}


extern "C" __global__ void rmsnorm(float *o, float *x, float *weight, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize;

    // ----- 1. Calculate partial sum of squares -----
    float partial_ss = 0.0f;
    for (int j = tid; j < size; j += blockDim.x) {
        partial_ss += x[j] * x[j];
    }

    // ----- 2. Warp Reduction (using shuffles) -----
    float warpSum = partial_ss;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      float otherValue = __shfl_down_sync(0xFFFFFFFF, warpSum, offset);
      warpSum += otherValue;
    }

    // ----- 3. Warp Leader Writes to Shared Memory -----
    extern __shared__ float shared[];
    if (laneId == 0) {
      shared[warpId] = warpSum;
    }
    __syncthreads();

    // ----- 4. Block Reduction (using shuffles) -----
    float blockSum = shared[laneId];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
      blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);

    // ----- 5. Broadcast the block sum (all threads do it)-----
    blockSum = __shfl_sync(0xFFFFFFFF, blockSum, 0);

    // ----- 6. Calculate global RMS normalization factor (all threads calculate it) -----
    float ss = blockSum;
    ss /= size;
    ss += 1e-5f;
    ss = rsqrtf(ss);

    // ----- 7. Normalize and scale each element -----
    for (int j = tid; j < size; j += blockDim.x) {
      o[j] = weight[j] * (ss * x[j]);
    }
}
