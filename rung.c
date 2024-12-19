/* Inference for Llama-3 Transformer model in pure C, targeting Nvidia RTX 3090 GPU with BF16 support */

#include <ctype.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

static CUfunction batched_softmax_kernel;
static CUfunction fp32_to_bf16_kernel;
static CUfunction swiGLU_kernel;
static CUfunction rope_rotary_encoding_kernel;
static CUfunction rmsnorm_kernel;
// Define USE_CUDA to enable CUDA GPU acceleration
#define USE_CUDA

// ----------------------------------------------------------------------------
// CUDA error checking
#define CHECK_CUDA(call)                                                                                                                                                           \
  do {                                                                                                                                                                             \
    cudaError_t err = (call);                                                                                                                                                      \
    if (err != cudaSuccess) {                                                                                                                                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                                                   \
      exit(EXIT_FAILURE);                                                                                                                                                          \
    }                                                                                                                                                                              \
  } while (0)

#define CHECK_CUBLAS(call)                                                                                                                                                         \
  do {                                                                                                                                                                             \
    cublasStatus_t status = (call);                                                                                                                                                \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                                                                                         \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);                                                                                                  \
      exit(EXIT_FAILURE);                                                                                                                                                          \
    }                                                                                                                                                                              \
  } while (0)

// ----------------------------------------------------------------------------
// BF16 Utilities

static inline float bf16_to_fp32(uint16_t bf16) {
  union {
    uint32_t u32;
    float fp32;
  } v;
  v.u32 = ((uint32_t)bf16) << 16;
  return v.fp32;
}

static inline uint16_t fp32_to_bf16(float fp32) {
  union {
    uint32_t u32;
    float fp32;
  } v;
  v.fp32 = fp32;
  return (uint16_t)(v.u32 >> 16);
}

// Utility function to convert FP32 to BF16
// We assume no subnormal numbers are passed
void fp32_to_bf16_array(uint16_t *output, float *input, size_t size) {
  if (input == NULL || output == NULL || size == 0) {
    return;
  }

  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 input_vec = _mm256_loadu_ps(input + i);

    // Do BF16 conversion
    __m256i int_vec = _mm256_castps_si256(input_vec);
    __m256i shifted_vec = _mm256_srli_epi32(int_vec, 16);
    __m128i upper_half = _mm256_extracti128_si256(shifted_vec, 1);
    __m128i lower_half = _mm256_extracti128_si256(shifted_vec, 0);
    __m128i bf16 = _mm_packus_epi32(lower_half, upper_half);

    _mm_storeu_si128((__m128i *)(output + i), bf16);
  }

  // Handle remaining elements (if size is not a multiple of 8)
  for (; i < size; ++i) {
    output[i] = fp32_to_bf16(input[i]);
  }
}

void bf16_to_fp32_array(float *out, uint16_t *in, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = bf16_to_fp32(in[i]);
  }
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
  int dim;        // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 4096 (byte-level)
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  uint16_t *wq; // (layer, dim, n_heads * head_size)
  uint16_t *wk; // (layer, dim, n_kv_heads * head_size)
  uint16_t *wv; // (layer, dim, n_kv_heads * head_size)
  uint16_t *wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  uint16_t *w1; // (layer, hidden_dim, dim)
  uint16_t *w2; // (layer, dim, hidden_dim)
  uint16_t *w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  uint16_t *wcls;
  // storage for the original fp32 weights
  float *rms_att_weight_fp32;
  float *rms_ffn_weight_fp32;
  float *wq_fp32;
  float *wk_fp32;
  float *wv_fp32;
  float *wo_fp32;
  float *w1_fp32;
  float *w2_fp32;
  float *w3_fp32;
  float *rms_final_weight_fp32;
  float *wcls_fp32;
} TransformerWeights;

typedef struct {
  // current wave of activations
  float *x;      // activation at current time stamp (dim,)
  float *xb;     // same, but inside a residual branch (dim,)
  float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q;      // query (dim,)
  float *k;      // key (dim,)
  float *v;      // value (dim,)
  float *att;    // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  void **ptrs;   // Device Pointers
  uint16_t *xb_bf16;
  uint16_t *hb_bf16;
  // kv cache
  float *key_cache;   // (layer, seq_len, dim)
  float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config;                  // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights;     // the weights of the model
  RunState state;                 // buffers for the "wave" of activations in the forward pass
  TransformerWeights weights_gpu; // GPU version of weights
  RunState state_gpu;             // GPU version of RunState
  cublasHandle_t handle;
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;            // file descriptor for memory mapping
  float *data;       // memory mapped data pointer
  ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void *calloc_aligned(size_t num, size_t size) {
  size_t total_size = num * size;

  void *ptr = NULL;
  if (posix_memalign(&ptr, 64, total_size) != 0) {
    return NULL;
  }

  memset(ptr, 0, total_size);

  return ptr;
}

void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->q = calloc(p->dim, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  s->ptrs = calloc(3 * 2 + 5 * p->n_heads, sizeof(void *));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void malloc_run_state_gpu(RunState *s, Config *p) {
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  CHECK_CUDA(cudaMalloc((void **)&s->x, p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->xb, p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->hb, p->hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->hb2, p->hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->q, p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->att, p->n_heads * p->seq_len * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->logits, p->vocab_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&s->ptrs, (3 * 2 + 5 * p->n_heads) * sizeof(float *)));
  CHECK_CUDA(cudaMalloc((void **)&s->xb_bf16, p->dim * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&s->hb_bf16, p->hidden_dim * sizeof(uint16_t)));
}

void free_run_state(RunState *s) {
  free(s->x);
  free(s->xb);
  free(s->hb);
  free(s->hb2);
  free(s->q);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
  free(s->ptrs);
}

void free_run_state_gpu(RunState *s) {
  if (s->x)
    CHECK_CUDA(cudaFree(s->x));
  if (s->xb)
    CHECK_CUDA(cudaFree(s->xb));
  if (s->hb)
    CHECK_CUDA(cudaFree(s->hb));
  if (s->hb2)
    CHECK_CUDA(cudaFree(s->hb2));
  if (s->q)
    CHECK_CUDA(cudaFree(s->q));
  if (s->key_cache)
    CHECK_CUDA(cudaFree(s->key_cache));
  if (s->value_cache)
    CHECK_CUDA(cudaFree(s->value_cache));
  if (s->att)
    CHECK_CUDA(cudaFree(s->att));
  if (s->logits)
    CHECK_CUDA(cudaFree(s->logits));
  if (s->ptrs)
    CHECK_CUDA(cudaFree(s->ptrs));
  if (s->xb_bf16)
    CHECK_CUDA(cudaFree(s->xb_bf16));
  if (s->hb_bf16)
    CHECK_CUDA(cudaFree(s->hb_bf16));
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  unsigned long long n_layers = p->n_layers;

  // Store the FP32 pointers
  w->token_embedding_table = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight_fp32 = ptr;
  ptr += n_layers * p->dim;
  w->wq_fp32 = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk_fp32 = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv_fp32 = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo_fp32 = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight_fp32 = ptr;
  ptr += n_layers * p->dim;
  w->w1_fp32 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2_fp32 = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3_fp32 = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight_fp32 = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2;
  ptr += p->seq_len * head_size / 2;
  w->wcls_fp32 = shared_weights ? w->token_embedding_table : ptr;
}

void malloc_weights_gpu(TransformerWeights *w, Config *p) {
  int head_size = p->dim / p->n_heads;
  unsigned long long n_layers = p->n_layers;
  // CHECK_CUDA(cudaMalloc((void **)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&w->rms_att_weight, n_layers * p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&w->wq, n_layers * p->dim * (p->n_heads * head_size) * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->wk, n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->wv, n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->wo, n_layers * (p->n_heads * head_size) * p->dim * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->rms_ffn_weight, n_layers * p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&w->w1, n_layers * p->dim * p->hidden_dim * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->w2, n_layers * p->hidden_dim * p->dim * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->w3, n_layers * p->dim * p->hidden_dim * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc((void **)&w->rms_final_weight, p->dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&w->wcls, p->vocab_size * p->dim * sizeof(uint16_t)));
}

void free_weights_gpu(TransformerWeights *w) {
  // CHECK_CUDA(cudaFree(w->token_embedding_table));
  CHECK_CUDA(cudaFree(w->rms_att_weight));
  CHECK_CUDA(cudaFree(w->wq));
  CHECK_CUDA(cudaFree(w->wk));
  CHECK_CUDA(cudaFree(w->wv));
  CHECK_CUDA(cudaFree(w->wo));
  CHECK_CUDA(cudaFree(w->rms_ffn_weight));
  CHECK_CUDA(cudaFree(w->w1));
  CHECK_CUDA(cudaFree(w->w2));
  CHECK_CUDA(cudaFree(w->w3));
  CHECK_CUDA(cudaFree(w->rms_final_weight));
  CHECK_CUDA(cudaFree(w->wcls));
}

void copy_weights_to_gpu(TransformerWeights *dest_gpu, TransformerWeights *src, Config *p) {
  int head_size = p->dim / p->n_heads;
  unsigned long long n_layers = p->n_layers;

  // Allocate temporary buffers for BF16 conversion
  uint16_t *temp_buffer;

  // 1. token_embedding_table
  size_t size = p->vocab_size * p->dim;
  // CHECK_CUDA(cudaMemcpy(dest_gpu->token_embedding_table, src->token_embedding_table, size * sizeof(float), cudaMemcpyHostToDevice));

  // 2. rms_att_weight
  size = n_layers * p->dim;
  CHECK_CUDA(cudaMemcpy(dest_gpu->rms_att_weight, src->rms_att_weight_fp32, size * sizeof(float), cudaMemcpyHostToDevice));

  // 3. wq
  size = n_layers * p->dim * (p->n_heads * head_size);
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->wq_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->wq, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 4. wk
  size = n_layers * p->dim * (p->n_kv_heads * head_size);
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->wk_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->wk, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 5. wv
  size = n_layers * p->dim * (p->n_kv_heads * head_size);
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->wv_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->wv, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 6. wo
  size = n_layers * (p->n_heads * head_size) * p->dim;
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->wo_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->wo, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 7. rms_ffn_weight
  size = n_layers * p->dim;
  CHECK_CUDA(cudaMemcpy(dest_gpu->rms_ffn_weight, src->rms_ffn_weight_fp32, size * sizeof(float), cudaMemcpyHostToDevice));

  // 8. w1
  size = n_layers * p->dim * p->hidden_dim;
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->w1_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->w1, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 9. w2
  size = n_layers * p->hidden_dim * p->dim;
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->w2_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->w2, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 10. w3
  size = n_layers * p->dim * p->hidden_dim;
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->w3_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->w3, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));

  // 11. rms_final_weight
  size = p->dim;
  CHECK_CUDA(cudaMemcpy(dest_gpu->rms_final_weight, src->rms_final_weight_fp32, size * sizeof(float), cudaMemcpyHostToDevice));

  // 12. wcls
  size = p->vocab_size * p->dim;
  CHECK_CUDA(cudaMallocHost((void **)&temp_buffer, size * sizeof(uint16_t)));
  fp32_to_bf16_array(temp_buffer, src->wcls_fp32, size);
  CHECK_CUDA(cudaMemcpy(dest_gpu->wcls, temp_buffer, size * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaFreeHost(temp_buffer));
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in the config header
  if (fread(config, sizeof(Config), 1, file) != 1) {
    exit(EXIT_FAILURE);
  }
  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  // figure out the file size
#if defined _WIN32
  _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = _ftelli64(file); // get the file size, in bytes
#else
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
#endif
  fclose(file);
  // memory map the Transformer weights into the data pointer
  *fd = open(checkpoint, O_RDONLY); // open in read only mode
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
  float *weights_ptr = *data + sizeof(Config) / sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
  // allocate and copy weights to GPU
  malloc_weights_gpu(&t->weights_gpu, &t->config);
  copy_weights_to_gpu(&t->weights_gpu, &t->weights, &t->config);
  // allocate GPU buffers
  malloc_run_state_gpu(&t->state_gpu, &t->config);
  // Create cublas handle
  CHECK_CUBLAS(cublasCreate(&t->handle));
  CHECK_CUBLAS(cublasSetMathMode(t->handle, CUBLAS_DEFAULT_MATH));
}

void free_transformer(Transformer *t) {
  // close the memory mapping
  if (t->data != MAP_FAILED) {
    munmap(t->data, t->file_size);
  }
  if (t->fd != -1) {
    close(t->fd);
  }
  // free the RunState buffers
  // free_run_state(&t->state);
  free_run_state_gpu(&t->state_gpu);
  free_weights_gpu(&t->weights_gpu);
  cublasDestroy(t->handle);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// Launch function for rmsnorm
void rmsnorm_gpu(float *d_o, float *d_x, float *d_weight, int size, cublasHandle_t cublas_handle) {
  void *args[] = {&d_o, &d_x, &d_weight, &size};

  // Configure kernel launch parameters
  int threads_per_block = 1024;
  int blocks_per_grid = 1;
  size_t sharedMemSize = (threads_per_block / 32) * sizeof(float);

  // Launch the kernel
  CHECK_CUDA(cuLaunchKernel(rmsnorm_kernel, blocks_per_grid, 1, 1, // grid dimensions
                            threads_per_block, 1, 1,               // block dimensions
                            sharedMemSize, NULL,                   // shared memory and stream
                            args, 0));                             // arguments
}

void batched_softmax_gpu(float *x, int size, int batch_size) {
  // Kernel parameters
  void *args[] = {&x, &size, &batch_size};

  // Kernel launch configuration
  int threadsPerBlock = 1024;
  int sharedMemSize = (threadsPerBlock / 32) * sizeof(float); // Shared memory for reductions
  int gridDim = batch_size;

  // Launch kernel
  CHECK_CUDA(cuLaunchKernel(batched_softmax_kernel, gridDim, 1, 1, // Grid dimensions
                            threadsPerBlock, 1, 1,                 // Block dimensions
                            sharedMemSize, 0,                      // Shared memory and stream
                            args, NULL));
}

void fp32_to_bf16_array_gpu(uint16_t *out, float *in, size_t n) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

  void *args[] = {&in, &out, &n};

  CHECK_CUDA(cuLaunchKernel(fp32_to_bf16_kernel, blocksPerGrid, 1, 1, // Grid dimensions
                            threadsPerBlock, 1, 1,                    // Block dimensions
                            0, 0,                                     // Shared memory and stream
                            args, NULL));
}

void swiGLU_gpu(float *hb, float *hb2, int hidden_dim) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (hidden_dim + threadsPerBlock - 1) / threadsPerBlock;

  void *args[] = {&hb, &hb2, &hidden_dim};

  CHECK_CUDA(cuLaunchKernel(swiGLU_kernel, blocksPerGrid, 1, 1, // Grid dimensions
                            threadsPerBlock, 1, 1,              // Block dimensions
                            0, 0,                               // Shared memory and stream
                            args, NULL));
}

// Function to launch the kernel
void rope_rotary_encoding_gpu(float *q_device, float *k_device, int n_heads, int n_kv_heads, int head_size, int pos) {
  // Define grid and block sizes
  int threadsPerBlock = head_size / 2; // Each thread processes two positions (j, j+1)
  int blocksPerGrid = n_heads;         // One block per attention head

  // Kernel arguments
  void *args[] = {&q_device, &k_device, &n_heads, &n_kv_heads, &head_size, &pos};

  // Launch the kernel
  CHECK_CUDA(cuLaunchKernel(rope_rotary_encoding_kernel, blocksPerGrid, 1, 1, // Grid dimensions
                            threadsPerBlock, 1, 1,                            // Block dimensions
                            0, 0,                                             // Shared memory and stream
                            args, NULL));
}

float *forward(Transformer *transformer, int token, int pos) {

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights_gpu;
  RunState *s = &transformer->state_gpu;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;
  cublasHandle_t handle = transformer->handle;

  const float one = 1.0f;
  const float zero = 0.0f;

  // copy the token embedding into x  -- **Corrected section**
  CHECK_CUDA(cudaMemcpy(x, transformer->weights.token_embedding_table + token * dim, dim * sizeof(float), cudaMemcpyHostToDevice));

  // Setup device memory pointers for batched operations
  void **ptrs = s->ptrs;
  float **q_pointers_d, **k_pointers_d, **v_pointers_d, **att_pointers_d, **xb_pointers_d;
  void **w_pointers_d, **xb_bf16_pointers_d, **h_pointers_d;

  q_pointers_d = (float **)&ptrs[p->n_heads * 0];
  k_pointers_d = (float **)&ptrs[p->n_heads * 1];
  v_pointers_d = (float **)&ptrs[p->n_heads * 2];
  att_pointers_d = (float **)&ptrs[p->n_heads * 3];
  xb_pointers_d = (float **)&ptrs[p->n_heads * 4];

  w_pointers_d = &ptrs[p->n_heads * 5 + 2 * 0];
  xb_bf16_pointers_d = &ptrs[p->n_heads * 5 + 2 * 1];
  h_pointers_d = &ptrs[p->n_heads * 5 + 2 * 2];

  // Setup host memory for device pointers
  void **ptrs_h = transformer->state.ptrs;
  float **q_pointers_h = (void *)&ptrs_h[p->n_heads * 0];
  float **k_pointers_h = (void *)&ptrs_h[p->n_heads * 1];
  float **v_pointers_h = (void *)&ptrs_h[p->n_heads * 2];
  float **att_pointers_h = (void *)&ptrs_h[p->n_heads * 3];
  float **xb_pointers_h = (void *)&ptrs_h[p->n_heads * 4];

  float **w_pointers_h = (void *)&ptrs_h[p->n_heads * 5 + 2 * 0];
  uint16_t **xb_bf16_pointers_h = (void *)&ptrs_h[p->n_heads * 5 + 2 * 1];
  float **h_pointers_h = (void *)&ptrs_h[p->n_heads * 5 + 2 * 2];

  float invsqrt_head_size = 1.0f / sqrtf(head_size);

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // Initialize the host pointers to point to the correct locations in the GPU memory
    for (int h = 0; h < p->n_heads; ++h) {
      q_pointers_h[h] = s->q + h * head_size;
      k_pointers_h[h] = s->key_cache + loff + (h / kv_mul) * head_size;
      v_pointers_h[h] = s->value_cache + loff + (h / kv_mul) * head_size;
      att_pointers_h[h] = s->att + h * (pos + 1);
      xb_pointers_h[h] = s->xb + h * head_size;
    }
    // Initialize host pointers for w1 and w3
    w_pointers_h[0] = (float *)(w->w1 + l * dim * hidden_dim);
    w_pointers_h[1] = (float *)(w->w3 + l * dim * hidden_dim);

    xb_bf16_pointers_h[0] = s->xb_bf16;
    xb_bf16_pointers_h[1] = s->xb_bf16;

    // Initialize host pointers for hb and hb2 (outputs)
    h_pointers_h[0] = s->hb;
    h_pointers_h[1] = s->hb2;

    // Copy the arrays of pointers from host to device
    CHECK_CUDA(cudaMemcpy(ptrs, ptrs_h, (2 * 3 + p->n_heads * 5) * sizeof(void *), cudaMemcpyHostToDevice));

    // attention rmsnorm
    rmsnorm_gpu(s->xb, x, w->rms_att_weight + l * dim, dim, handle);

    fp32_to_bf16_array_gpu(s->xb_bf16, s->xb, dim);

    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim, dim, &one, s->xb_bf16, CUDA_R_16BF, 1, w->wq + l * dim * dim, CUDA_R_16BF, dim, &zero, s->q, CUDA_R_32F, 1,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, kv_dim, dim, &one, s->xb_bf16, CUDA_R_16BF, 1, w->wk + l * dim * kv_dim, CUDA_R_16BF, dim, &zero, s->k,
                              CUDA_R_32F, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, kv_dim, dim, &one, s->xb_bf16, CUDA_R_16BF, 1, w->wv + l * dim * kv_dim, CUDA_R_16BF, dim, &zero, s->v,
                              CUDA_R_32F, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    rope_rotary_encoding_gpu(s->q, s->k, p->n_heads, p->n_kv_heads, head_size, pos);

    // 2. Multiply Q by K^T for each head to get attention scores
    CHECK_CUBLAS(cublasGemmBatchedEx(transformer->handle, CUBLAS_OP_T, CUBLAS_OP_N, pos + 1, 1, head_size, &invsqrt_head_size, (const void *const *)k_pointers_d, CUDA_R_32F,
                                     kv_dim, (const void *const *)q_pointers_d, CUDA_R_32F, head_size, &zero, (void *const *)att_pointers_d, CUDA_R_32F, pos + 1, p->n_heads,
                                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

    batched_softmax_gpu(s->att, pos + 1, p->n_heads);

    // 4. Multiply each attention matrix by V
    CHECK_CUBLAS(cublasGemmBatchedEx(transformer->handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, 1, pos + 1, &one, (const void *const *)v_pointers_d, CUDA_R_32F, kv_dim,
                                     (const void *const *)att_pointers_d, CUDA_R_32F, pos + 1, &zero, (void *const *)xb_pointers_d, CUDA_R_32F, head_size, p->n_heads, CUDA_R_32F,
                                     CUBLAS_GEMM_DEFAULT));

    // final matmul to get the output of the attention
    fp32_to_bf16_array_gpu(s->xb_bf16, s->xb, dim);
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim, dim, &one, s->xb_bf16, CUDA_R_16BF, 1, w->wo + l * dim * dim, CUDA_R_16BF, dim, &one, x, CUDA_R_32F, 1,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // ffn rmsnorm
    rmsnorm_gpu(s->xb, x, w->rms_ffn_weight + l * dim, dim, handle);

    fp32_to_bf16_array_gpu(s->xb_bf16, s->xb, dim);

    // --- Perform batched matrix multiplication ---
    CHECK_CUBLAS(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, hidden_dim, dim, &one, (const void *const *)xb_bf16_pointers_d, CUDA_R_16BF, 1,
                                     (const void *const *)w_pointers_d, CUDA_R_16BF, dim, &zero, (void *const *)h_pointers_d, CUDA_R_32F, 1, 2, CUDA_R_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    swiGLU_gpu(s->hb, s->hb2, p->hidden_dim);

    fp32_to_bf16_array_gpu(s->hb_bf16, s->hb, hidden_dim);

    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, dim, hidden_dim, &one, s->hb_bf16, CUDA_R_16BF, 1, w->w2 + l * dim * hidden_dim, CUDA_R_16BF, hidden_dim, &one,
                              x, CUDA_R_32F, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  // final rmsnorm
  rmsnorm_gpu(x, x, w->rms_final_weight, dim, handle);

  // classifier into logits
  fp32_to_bf16_array_gpu(s->xb_bf16, x, dim);
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, p->vocab_size, dim, &one, s->xb_bf16, CUDA_R_16BF, 1, w->wcls, CUDA_R_16BF, dim, &zero, s->logits, CUDA_R_32F, 1,
                            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); }

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char **)malloc(vocab_size * sizeof(char *));
  t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", tokenizer_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
    exit(EXIT_FAILURE);
  }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {
      fprintf(stderr, "failed read\n");
      exit(EXIT_FAILURE);
    }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer *t) {
  for (int i = 0; i < t->vocab_size; i++) {
    free(t->vocab[i]);
  }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];

  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = {.str = str}; // acts as the key to search for
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) {
    fprintf(stderr, "cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=128000) token, if desired
  if (bos)
    tokens[(*n_tokens)++] = 128000;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i = 0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair or triple each iteration, according to the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    int best_len = 2; // length of the best merge sequence (2 for pair, 3 for triple)

    // first, try to find the best pair to merge
    for (int i = 0; i < (*n_tokens - 1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    // if no pair was found, try to find the best triple to merge
    if (best_idx == -1) {
      for (int i = 0; i < (*n_tokens - 2); i++) {
        // check if we can merge the triple (tokens[i], tokens[i+1], tokens[i+2])
        sprintf(str_buffer, "%s%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]], t->vocab[tokens[i + 2]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
          // this merge triple exists in vocab! record its score and position
          best_score = t->vocab_scores[id];
          best_id = id;
          best_idx = i;
          best_len = 3;
        }
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs or triples to merge, so we're done
    }

    // merge the consecutive pair or triple (best_idx, best_idx+1[, best_idx+2]) into new token best_id
    tokens[best_idx] = best_id;
    // delete token(s) at position best_idx+1 (and optionally best_idx+2), shift the entire sequence back
    for (int i = best_idx + 1; i < (*n_tokens - best_len + 1); i++) {
      tokens[i] = tokens[i + best_len - 1];
    }
    (*n_tokens) -= (best_len - 1); // token length decreased by the number of merged tokens minus one
  }

  // add optional EOS (=128001) token, if desired
  if (eos)
    tokens[(*n_tokens)++] = 128001;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob)
    return -1;
  if (a_->prob < b_->prob)
    return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
      logits[q] /= sampler->temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;               // used to time our code, only initialized after first iteration
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      float *logits_cpu = (float *)malloc(transformer->config.vocab_size * sizeof(float));
      CHECK_CUDA(cudaMemcpy(logits_cpu, logits, transformer->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
      next = sample(sampler, logits_cpu);
      free(logits_cpu);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
      break;
    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are somewhat haphazardly and unsafely set atm
  char *system_prompt = (char *)malloc(32768 * sizeof(char));
  char *user_prompt = (char *)malloc(32768 * sizeof(char));
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *system_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int *user_prompt_tokens = (int *)malloc(32768 * sizeof(int));
  int user_idx = 0;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;             // will store the next token in the sequence
  int token;            // stores the current token to feed into the transformer

  int pos = 0; // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
        prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 9125;   // "system"
        prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
        prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, 32768);
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
        if (system_prompt != NULL) {
          int num_system_prompt_tokens = 0;
          encode(tokenizer, system_prompt, 0, 0, system_prompt_tokens, &num_system_prompt_tokens);
          for (int i = 0; i < num_system_prompt_tokens; i++) {
            prompt_tokens[num_prompt_tokens++] = system_prompt_tokens[i];
          }
        }
        prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      } else {
        num_prompt_tokens = 0;
      }
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 882;    // "user"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User (or exit): ", user_prompt, 32768);
        if (strcmp(user_prompt, "exit") == 0)
          break;
      }
      int num_user_prompt_tokens = 0;
      // encode the user prompt into tokens
      encode(tokenizer, user_prompt, 0, 0, user_prompt_tokens, &num_user_prompt_tokens);
      for (int i = 0; i < num_user_prompt_tokens; i++) {
        prompt_tokens[num_prompt_tokens++] = user_prompt_tokens[i];
      }
      prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
      prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
      prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
      prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=128009) token ends the Assistant turn
    if (user_idx >= num_prompt_tokens && (token == 128009 || token == 128001)) {
      user_turn = 1;
    }

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    float *logits_cpu = (float *)malloc(transformer->config.vocab_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(logits_cpu, logits, transformer->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    next = sample(sampler, logits_cpu);
    free(logits_cpu);

    pos++;

    if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 && next != 128006) {
      // the Assistant is responding, so print its output
      char *piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (user_idx >= num_prompt_tokens && next == 128009 || next == 128001) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
  free(system_prompt_tokens);
  free(user_prompt_tokens);
  free(system_prompt);
  free(user_prompt);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *checkpoint_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 4096;                // number of steps to run for
  char *prompt = NULL;             // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";         // generate|chat
  char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0)
    rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
  if (steps < 0)
    steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len)
    steps = transformer.config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  CUcontext current_context;
  CHECK_CUDA(cuCtxGetCurrent(&current_context));
  CUmodule cuModule;
  CHECK_CUDA(cuModuleLoad(&cuModule, "rung.ptx"));
  CHECK_CUDA(cuModuleGetFunction(&batched_softmax_kernel, cuModule, "batched_softmax"));
  CHECK_CUDA(cuModuleGetFunction(&fp32_to_bf16_kernel, cuModule, "fp32_to_bf16"));
  CHECK_CUDA(cuModuleGetFunction(&swiGLU_kernel, cuModule, "swiGLU"));
  CHECK_CUDA(cuModuleGetFunction(&rope_rotary_encoding_kernel, cuModule, "rope_rotary_encoding"));
  CHECK_CUDA(cuModuleGetFunction(&rmsnorm_kernel, cuModule, "rmsnorm"));

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
