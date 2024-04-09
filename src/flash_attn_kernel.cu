#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>

#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {
/**
@brief: softmax_kernel
Softmax forward kernel for
  enc-self-attn, dec-self-attn, encdec-attn

@thread
gridDim.x = dynamic
gridDim.y = batch_size
gridDim.z = nhead
blockDim.x = from_len

@param
inp: [batch_size, nhead, from_len, to_len], softmax input.
attn_mask: [batch_size, to_len], padding tokens are -inf,
  non padding tokens are 0.
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(T *inp, const T *attn_mask, int from_len,
                                      int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    // Hint: use fmaxf() to compute max
    // BEGIN ASSIGN3_1
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        float temp_val;
        if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
          temp_val = REDUCE_FLOAT_INF_NEG;
        } else {
          temp_val = (float)inp_val[i][j];
          //printf("%f\n",(float)mval[j]);
          if (attn_mask) {
            temp_val += (float)mval[j];
          }
        }
        val[i][j] = temp_val;
        l_max[i] = fmaxf(l_max[i], temp_val);
      }
    }
    // END ASSIGN3_1
    // warp reduce max
    warpReduce<ReduceType::kMax, token_per_reduce>(l_max);

    /* step 2. compute sum */
    // thread local sum
    // BEGIN ASSIGN3_1
    // Hint: use __expf() to compute exp
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - l_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // END ASSIGN3_1
    // warp reduce sum
    warpReduce<ReduceType::kSum, token_per_reduce>(l_sum);

    /* step 3. compute final result */
    // BEGIN ASSIGN3_1
    // Hint: use __fdividef() to compute division
    // Hint: use BlockStore to store the result
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      l_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * l_sum[i]);
      }
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
    // END ASSIGN3_1
  }  // blockIdx.x
}

__global__ void FlashAttnKernel(
    float* out,
    float* Q,
    float* K,
    float* V,
    float* m,
    float* l,
    const int N, 
    const int d,
    const int B_r, 
    const int B_c
) {

    int j = threadIdx.y;
    int i = threadIdx.x;

    float* Q_i = new float[B_r * B_r];
    float* m_i = new float[B_r];
    float* l_i = new float[B_r];
    float* O_i = new float[B_r * B_r];

    int row, col;
    for (row = i * B_r; row < (i + 1) * B_r; row++) {
        for (col = 0; col < d; col++) {
            Q_i[(row - i * B_r) * d + col] = Q[row * d + col];
            O_i[(row - i * B_r) * d + col] = out[row * d + col];
        }
        m_i[(row - i * B_r)] = m[row];
        l_i[(row - i * B_r)] = l[row];
    }

    float* S_ij = new float[B_r * B_c];
    float* m_ij = new float[B_r];
    float* l_ij = new float[B_r];
    float* m_new = new float[B_r];

    float sum;
    float max;

    for (int row = 0; row < B_r; row++) {
      max = 0;
      for (int col = 0; col < B_c; col++) {
          sum = 0;
          for (int k = 0; k < B_r; k++) { 
              int indexA = row * d + k; 
              int indexB = col * d + k; 
              sum += Q_i[indexA] * K[indexB];
          }
          S_ij[row * B_c + col] = sum; 
          if (sum > max) {
            max = sum;
          }
      }
      m_ij[row] = max;
      m_new[row] = (m_ij[row] > m_i[row]) ? m_ij[row] : m_i[row];
    }

    for (int row = 0; row < B_r; row++) {
      sum = 0;
      for (int col = 0; col < B_c; col++) {
        float val = exp(S_ij[row * B_c + col] - m_ij[row]);
        S_ij[row * B_c + col] = val;
        sum += val;
      }
      l_ij[row] = sum;
    }

    float* l_new = new float[B_r];
    
    for (int row = 0; row < B_r; row++) {
      l_new[row] = exp(m_i[row] - m_new[row]) * l_i[row] + exp(m_ij[row] - m_new[row]) * l_ij[row];
    }

    for (int row = 0; row < B_r; row++) {
      for (int col = 0; col < B_r; col++) {
          sum = 0;
          for (int k = 0; k < N; k++) { 
              int indexA = row * N + k; 
              int indexB = k * B_r + col; 
              sum += S_ij[indexA] * V[indexB];
          }
          Q_i[row * B_r + col] = sum * exp(m_ij[row] - m_new[row]); 
      }
    }

    for (int row = 0; row < B_r; row++) {
      for (int col = 0; col < B_r; col++) {
          sum = 0;
          for (int k = 0; k < N; k++) { 
              int indexA = row * N + k; 
              int indexB = k * B_r + col; 
              sum += S_ij[indexA] * V[indexB];
          }
          Q_i[row * B_r + col] = sum * exp(m_ij[row] - m_new[row]); 
      }
    }


    int t = 0;
    for (int row = 0; row < B_r; row++) {
      for (int col = 0; col < B_r; col++) {
          sum = 0;
          for (int k = 0; k < B_r; k++) { 
              int indexA = row * B_r + k; 
              int indexB = k * B_r + col; 
              if (indexA % (B_r + 1) == 0) {
                if (i == 0) {
                  printf("%d \n", indexA);
                }
                sum += (1 / l_new[t]) * K[indexB];
                t += 1;
              }
          }
          O_i[row * B_r + col] = sum;
      }
    }

    /// END ASSIGN1_2
}

/*
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
// template <>
extern "C" {
void launch_attn_flash(float *out, float *Q, float *K, float *V, 
                        float *m, float *l, const int N, const int d, const int B_r, const int B_c, const int T_r, const int T_c, cudaStream_t stream) {


  int float_size = sizeof(float);
  int inp_matrix_size = N * d * float_size;
  int inp_vector_size = N * float_size;

  float *d_out, *d_Q, *d_K, *d_V, *d_m, *d_l;
  cudaMalloc((void **)&d_out, inp_matrix_size);
  cudaMalloc((void **)&d_Q, inp_matrix_size);
  cudaMalloc((void **)&d_K, inp_matrix_size);
  cudaMalloc((void **)&d_V, inp_matrix_size);
  cudaMalloc((void **)&d_m, inp_vector_size);
  cudaMalloc((void **)&d_l, inp_vector_size);

  cudaMemcpy(d_out, out, inp_matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q, Q, inp_matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, K, inp_matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, inp_matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m, inp_vector_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l, l, inp_vector_size, cudaMemcpyHostToDevice);

  printf("cuda vals %d %d %d %d \n", B_r, B_c, T_r, T_c);
  
  dim3 blockDims(T_r, T_c);
  dim3 gridDims(1, 1, 1);
  FlashAttnKernel<<<gridDims, blockDims>>>(
        d_out, d_Q, d_K, d_V, d_m, d_l, N, d, B_r, B_c
  );
  
  // if (to_len <= 32) {
  //   ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else if (to_len <= 64) {
  //   ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else if (to_len <= 128) {
  //   grid_dim.x = 16;
  //   ker_attn_softmax<float, 64, 2><<<grid_dim, 64, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else if (to_len <= 256) {
  //   grid_dim.x = 32;
  //   ker_attn_softmax<float, 128, 2><<<grid_dim, 128, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else if (to_len <= 512) {
  //   grid_dim.x = 64;
  //   ker_attn_softmax<float, 256, 2><<<grid_dim, 256, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else if (to_len <= 1024) {
  //   grid_dim.x = 128;
  //   ker_attn_softmax<float, 512, 2><<<grid_dim, 512, 0, stream>>>(
  //       d_inp, d_attn_mask, from_len, to_len, mask_future);
  // } else {
  //   throw std::runtime_error(
  //       "Sequence length greater than 512 is currently not supported");
  // }

  // // Copy back to the host
  // cudaMemcpy(inp, d_inp, inp_size, cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();

  // // Check CUDA execution
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   fprintf(stderr, "launch_attn_softmax Error: %s\n", cudaGetErrorString(err));
  //   exit(EXIT_FAILURE);
  // }

  // // Free memory on device
  // cudaFree(d_inp);
  // cudaFree(d_attn_mask);

}}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


}  
} 
