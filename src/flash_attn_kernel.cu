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
