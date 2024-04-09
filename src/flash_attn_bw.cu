#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cmath>
#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>
#define BASE_THREAD_NUM 16
#define TILE_SIZE 256

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {

template <typename T> //, int block_dim, int ele_per_thread>
__global__ void flash_attn_bw(T *q, T *k, T *v, T *out, T *out_grad, T* q_grad, T* k_grad, T* v_grad, T *l, T *m, int batch, int N, int d) {
    int batch_idx = blockIdx.x;
    int tidx = threadIdx.x;
    int tidx_y = threadIdx.y;
    
    //printf("Batch Idx : %d\n", batch_idx);
    q += batch_idx * N * d;
    k +=  batch_idx * N * d;
    v += batch_idx * N * d;
    out_grad += batch_idx * N * d;
    out += batch_idx * N * d;
    q_grad += batch_idx * N * d;
    k_grad += batch_idx * N * d;
    v_grad += batch_idx * N * d;
    l += batch_idx * N;
    m += batch_idx * N;
    
    float tau = sqrt(1.0/d);
    int on_chip_memory_size = d * 256;
    int B_c = BASE_THREAD_NUM; //on_chip_memory_size / (4 * d);  // Using 4 bytes per float
    int B_r = min(BASE_THREAD_NUM, d); //min(on_chip_memory_size / (4 * d), d);
    int T_r = (N + B_r - 1)/ B_r;
    int T_c = (N +B_c -1)/ B_c;
    
    int tile_size = B_c * d; 
    
    assert(d < TILE_SIZE/BASE_THREAD_NUM);
    __shared__ float sram[TILE_SIZE * 12];
    float* Qi = sram;
    float* Kj = &sram[TILE_SIZE];
    float* Vj = &sram[TILE_SIZE * 2];
    float* Sij  = &sram[TILE_SIZE * 3];

    float* dQi = &sram[TILE_SIZE * 4];
    float* dKj = &sram[TILE_SIZE * 5];
    float* dVj = &sram[TILE_SIZE * 6];
    float* Pij  = &sram[TILE_SIZE * 7];
    float* dPij  = &sram[TILE_SIZE * 8];
    float* Oi = &sram[TILE_SIZE * 9];
    float* dOi  = &sram[TILE_SIZE * 10];
    float* dSij  = &sram[TILE_SIZE * 11];
    
    __shared__ float lm_sram[TILE_SIZE * 3];
    float* li = lm_sram;
    float* mi = &lm_sram[TILE_SIZE];
    float* Di = &lm_sram[TILE_SIZE * 2];
    


    for(int j = 0; j < T_c; j++){
        // Loading
        for(int y = 0; y < d; y++){
            if(tidx < B_c && j * B_c + tidx < N && tidx_y == 0){
                Kj[tidx * d + y] = k[(j * B_c + tidx) * d + y];
                Vj[tidx * d + y] = v[(j * B_c + tidx) * d + y];
                dKj[tidx * d + y] = 0;
                dVj[tidx * d + y] = 0;
            }
            else if(tidx_y == 0){
                Kj[tidx * d + y] = 0;
                Vj[tidx * d + y] = 0;
                dKj[tidx * d + y] = 0;
                dVj[tidx * d + y] = 0;
            }
        }
        for(int i = 0; i < T_r; i++){
            // Loading 
            for(int y = 0; y < d; y++){
                if(tidx < B_r && i * B_r + tidx < N && tidx_y == 0){
                    //assert(tidx * B_c + y < B_r);
                    Oi[tidx * d + y]  = out[(i * B_r + tidx) * d + y];
                    dOi[tidx * d + y] = out_grad[(i * B_r + tidx) * d + y];
                    Qi[tidx * d + y]  = q[(i * B_r + tidx) * d + y];
                    dQi[tidx * d + y] = q_grad[(i * B_r + tidx) * d + y];
                    
                    
                    if(y==0){
                        li[tidx] = l[i * B_r + tidx];
                        mi[tidx] = m[i * B_r + tidx];
                        Di[tidx] = 0;
                    }
                }
                else if(tidx_y == 0){
                    Oi[tidx * d + y]  = 0;
                    dOi[tidx * d + y] = 0;
                    Qi[tidx * d + y]  = 0;
                    dQi[tidx * d + y] = 0;
                }
            }
            
            Sij[tidx * B_c + tidx_y] = 0;
            dSij[tidx * B_c + tidx_y] = 0;
            Pij[tidx * B_c + tidx_y] = 0;
            dPij[tidx * B_c + tidx_y] = 0;
            
            
            __syncthreads();
            for(int y = 0; y < d; y++){
                if(tidx <B_r && i * B_r + tidx < N && tidx_y < B_c && j * B_c + tidx_y < N){
                //if(tidx <B_r && tidx_y < B_c){
                    Sij[tidx * B_c + tidx_y] += (tau * Qi[tidx * d + y] * Kj[tidx_y * d + y]);
                }   
            }
            __syncthreads();
            if(tidx <B_r && i * B_r + tidx < N && tidx_y < B_c && j * B_c + tidx_y < N){ 
            //if(tidx < B_r && tidx_y < B_c){  
                Pij[tidx * B_c + tidx_y] = (1.0/li[tidx]) * exp(Sij[tidx * B_c + tidx_y] - mi[tidx]);
            }  
            __syncthreads();
            for(int y = 0; y < max(B_r, d); y++){
                if(tidx < B_c && tidx_y < d && y < B_r){
                    dVj[tidx * d + tidx_y] += (Pij[y * B_c + tidx] * dOi[y * d + tidx_y]);
                }   
                if(tidx < B_r && tidx_y < B_c && y < d){
                    dPij[tidx * B_c + tidx_y] += (dOi[tidx * d + y] * Vj[tidx_y * d + y]);
                }   
                if(tidx < B_r && y < d){
                    if(tidx_y == 0)
                        Di[tidx] += Oi[tidx * d + y] * dOi[tidx * d + y];
                }   
            }
            __syncthreads();
            //if(tidx < B_r && tidx_y < B_c){
            if(tidx <B_r && i * B_r + tidx < N && tidx_y < B_c && j * B_c + tidx_y < N){ 
                dSij[tidx * B_c + tidx_y] = Pij[tidx * B_c + tidx_y] * (dPij[tidx * B_c + tidx_y] - Di[tidx]);
            }  
            __syncthreads();
            if(tidx < B_r && tidx_y < d){
                for(int y = 0; y < B_c && j * B_c + y < N; y++){
                    dQi[tidx * d + tidx_y] += (tau * dSij[tidx * B_c + y] * Kj[y * d + tidx_y]);
                }   
            }
            __syncthreads();
            for(int y = 0; y < max(B_r, d); y++){
                if(tidx < B_c && tidx_y < d && y < B_r){
                    dKj[tidx * d + tidx_y] += (tau * dSij[y * B_c + tidx] * Qi[y * d + tidx_y]);    
                }   
                if(tidx < B_r && i * B_r + tidx < N && tidx_y == 0 && y < d){
                    float old = q_grad[i * d * B_r + tidx * d + y];
                    q_grad[i * d * B_r + tidx * d + y] = dQi[tidx * d + y];
                }
            }
            __syncthreads();
        }
        for(int y = 0; y < d; y++){
            if(tidx < B_c && j * B_c + tidx < N && tidx_y == 0){
                k_grad[j * d * B_c + tidx * d + y] = dKj[tidx * d + y];
                v_grad[j * d * B_c + tidx * d + y] = dVj[tidx * d + y];
            }
        }
        __syncthreads();

    }
}



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" {

void launch_flash_attn_bw(
    float* q,
    float* k,
    float* v,
    float* out,
    float* out_grad,
    float* q_grad,
    float* k_grad,
    float* v_grad,
    float* l,
    float* m,
    int batch, int N, int d,
    cudaStream_t stream
) {
    
    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_out, *d_out_grad, *d_q_grad, *d_k_grad, *d_v_grad, *d_l, *d_m;
    cudaMalloc(&d_q, batch * N * d * sizeof(float));
    cudaMalloc(&d_k, batch * N * d * sizeof(float));
    cudaMalloc(&d_v, batch * N * d * sizeof(float));
    cudaMalloc(&d_out, batch * N * d * sizeof(float));
    cudaMalloc(&d_out_grad, batch * N * d * sizeof(float));
    cudaMalloc(&d_q_grad, batch * N * d * sizeof(float));
    cudaMalloc(&d_k_grad, batch * N * d * sizeof(float));
    cudaMalloc(&d_v_grad, batch * N * d * sizeof(float));
    
    cudaMalloc(&d_l, batch * N * sizeof(float));
    cudaMalloc(&d_m, batch * N * sizeof(float));
    

    // Copy data to the device
    cudaMemcpy(d_q, q, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_grad, out_grad, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_grad, q_grad, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_grad, k_grad, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_grad, v_grad, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, batch * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, batch * N * sizeof(float), cudaMemcpyHostToDevice);
    

    //std::cout << "B_c: " << B_c << "|B_r:" << B_r << "|tau:" << tau << "\n";
    
    dim3 grid_dim(batch);  // batch_size x num_heads
    dim3 block_dim(BASE_THREAD_NUM, BASE_THREAD_NUM);

    flash_attn_bw<float><<<grid_dim, block_dim, 0, stream>>>(d_q, d_k, d_v, d_out, d_out_grad, d_q_grad, d_k_grad, d_v_grad, d_l, d_m, batch, N, d);
      //      d_out_grad, d_soft_inp, softmax_len);
    /*
    //int threadsPerBlock = BASE_THREAD_NUM;
    //dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    //dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    //MatrixMultiplyKernel<<<gridDims, blockDims>>>(
    //    d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides
    //);

    //ker_attn_softmax_bw<float, 2048/WARP_SIZE><<<grid_dim, block_dim, 0, stream>>>(
      //      d_out_grad, d_soft_inp, softmax_len);
    // Copy back to the host
    */
    cudaMemcpy(q_grad, d_q_grad, batch * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_grad, d_k_grad, batch * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_grad, d_v_grad, batch * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Flash Attention Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_out_grad);
    cudaFree(d_q_grad);
    cudaFree(d_k_grad);
    cudaFree(d_v_grad);
    cudaFree(d_l);
    cudaFree(d_m);
    
}

}
}  
} 
