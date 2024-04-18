#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cmath>
#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>
#define BASE_THREAD_NUM 32
#define TILE_SIZE 1024

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {

template <typename T> //, int block_dim, int ele_per_thread>
__global__ void flash_attn_fw(T *q, T *k, T *v, T *out, T *l, T *m, int batch, int N, int d) {
    int batch_idx = blockIdx.x;
    int tidx = threadIdx.x;
    int tidx_y = threadIdx.y;
    
    //printf("Batch Idx : %d\n", batch_idx);
    q += batch_idx * N * d;
    k +=  batch_idx * N * d;
    v += batch_idx * N * d;
    out += batch_idx * N * d;
    l += batch_idx * N;
    m += batch_idx * N;
    
    /*if(batch_idx == 1 && tidx == 0 && tidx_y == 0){
        printf("Batch %d: %f, %f, %f\n", batch_idx, q[0], k[0], v[0]);
        printf("Batch %d: %f, %f, %f, %f\n", batch_idx, out_grad[0], out[0], l[0], m[0]);
        printf("Batch %d: %f, %f, %f\n", batch_idx, q_grad[0], k_grad[0], v_grad[0]);
    }*/
    //printf("%f\n", l[0]);
    
    float tau = sqrt(1.0/d);
    int on_chip_memory_size = d * 256;
    int B_c = BASE_THREAD_NUM; //on_chip_memory_size / (4 * d);  // Using 4 bytes per float
    int B_r = min(BASE_THREAD_NUM, d); //min(on_chip_memory_size / (4 * d), d);
    int T_r = (N + B_r - 1)/ B_r;
    int T_c = (N +B_c -1)/ B_c;

    int tile_size = B_c * d; 
    
    assert(d < TILE_SIZE/BASE_THREAD_NUM);
    __shared__ float sram[TILE_SIZE * 5];
    float* Qi = sram;
    float* Kj = &sram[TILE_SIZE];
    float* Vj = &sram[TILE_SIZE * 2];
    float* Sij  = &sram[TILE_SIZE * 3];
    float* tempPRO  = &sram[TILE_SIZE * 4];

    __shared__ float lm_sram[BASE_THREAD_NUM * 6];
    float* li = lm_sram;
    float* mi = &lm_sram[BASE_THREAD_NUM];
    float* lij = &lm_sram[BASE_THREAD_NUM * 2];
    float* mij = &lm_sram[BASE_THREAD_NUM * 3];
    float* lnew = &lm_sram[BASE_THREAD_NUM * 4];
    float* mnew = &lm_sram[BASE_THREAD_NUM * 5];
    


    for(int j = 0; j < T_c; j++){
        // Loading
        for(int y = 0; y < d; y++){
            if(tidx < B_c && j * B_c + tidx < N && tidx_y == 0){
                Kj[tidx * d + y] = k[(j * B_c + tidx) * d + y];
                Vj[tidx * d + y] = v[(j * B_c + tidx) * d + y];
            }
            else if(tidx_y == 0){
                Kj[tidx * d + y] = 0;
                Vj[tidx * d + y] = 0;
            }
        }
        for(int i = 0; i < T_r; i++){
            // Loading 
            for(int y = 0; y < d; y++){
                if(tidx < B_r && i * B_r + tidx < N && tidx_y == 0){
                    //assert(tidx * B_c + y < B_r);
                    Qi[tidx * d + y]  = q[(i * B_r + tidx) * d + y];
                    tempPRO[tidx * d + y] = 0;
                    if(y==0){
                        li[tidx] = l[i * B_r + tidx];
                        lij[tidx] = 0;
                        mi[tidx] = m[i * B_r + tidx];
                        mij[tidx] = -10000000;
                    }
                    
                }
                else if(tidx_y == 0){
                    Qi[tidx * d + y]  = 0;
                    tempPRO[tidx * d + y] = 0; 
                }
            }
            
            Sij[tidx * B_c + tidx_y] = 0;

            __syncthreads();
            for(int y = 0; y < d; y++){
                if(tidx <B_r && (i * B_r + tidx < N) && tidx_y < B_c && (j * B_c + tidx_y < N)){
                    Sij[tidx * B_c + tidx_y] += (tau * Qi[tidx * d + y] * Kj[tidx_y * d + y]);
                }   
            }
            __syncthreads();
            for(int y = 0; y < B_c; y++){
                if(tidx < B_r && (i * B_r + tidx < N) && tidx_y == 0){
                    mij[tidx] = max(mij[tidx], Sij[tidx * B_c + y]);
                }  
            }
            __syncthreads();
            if(tidx <B_r && (i * B_r + tidx < N) && tidx_y < B_c && (j * B_c + tidx_y < N)){
                Sij[tidx * B_c + tidx_y] = exp(Sij[tidx * B_c + tidx_y] - mij[tidx]);
            } 
            __syncthreads();
            for(int y = 0; y < B_c; y++){
                if(tidx < B_r && (i * B_r + tidx < N) && tidx_y == 0){
                    lij[tidx] += Sij[tidx * B_c + y];
                }  
            }
            __syncthreads();
            if(tidx < B_r && (i * B_r + tidx < N) && tidx_y == 0){
                mnew[tidx] = max(mi[tidx], mij[tidx]);
                lnew[tidx] = li[tidx] * exp(mi[tidx] - mnew[tidx]) + lij[tidx] * exp(mij[tidx] - mnew[tidx]);
            }  
            __syncthreads();

            for(int y = 0; y < B_c; y++){
                if(tidx < B_r && (i * B_r + tidx < N) && tidx_y < d){
                    tempPRO[tidx * d + tidx_y] += (Sij[tidx * B_c + y] * Vj[y * d + tidx_y]);
                }   
            }
            __syncthreads();

            for(int y = 0; y < d; y++){
                if(tidx < B_r && i * B_r + tidx < N && tidx_y == 0){
                    out[(i * B_r + tidx) * d + y] = (1.0/lnew[tidx]) * (li[tidx] * exp(mi[tidx] - mnew[tidx]) * out[(i * B_r + tidx) * d + y] + exp(mij[tidx] - mnew[tidx]) * tempPRO[tidx * d + y]);
                    
                    if(y==0){
                        l[i * B_r + tidx] = lnew[tidx];
                        m[i * B_r + tidx] = mnew[tidx]; 
                    }
                }
            }
            __syncthreads();
            /*
            if(tidx == 0 && tidx_y == 0 && batch_idx == 0){

                printf("\Pij cuda\n");
                for(int x = 0;x<B_r;x++){
                    for(int y = 0;y< B_c;y++){
                        printf("%f ", Pij[x * B_c + y]);
                    }
                    printf("\n");
                }
                
                printf("lnew cuda\n");
                for(int x = 0;x<B_r;x++){
                    printf("%f ", mij[x]);
                }
                printf("\n\n");

            }*/
            __syncthreads();
            
        }
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

void launch_flash_attn_fw(
    float* q,
    float* k,
    float* v,
    float* out,
    float* l,
    float* m,
    int batch, int N, int d,
    cudaStream_t stream
) {
    
    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_out, *d_l, *d_m;
    cudaMalloc(&d_q, batch * N * d * sizeof(float));
    cudaMalloc(&d_k, batch * N * d * sizeof(float));
    cudaMalloc(&d_v, batch * N * d * sizeof(float));
    cudaMalloc(&d_out, batch * N * d * sizeof(float));    
    cudaMalloc(&d_l, batch * N * sizeof(float));
    cudaMalloc(&d_m, batch * N * sizeof(float));
    

    // Copy data to the device
    cudaMemcpy(d_q, q, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, batch * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, batch * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, batch * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(batch);  // batch_size x num_heads
    dim3 block_dim(BASE_THREAD_NUM, BASE_THREAD_NUM);

    flash_attn_fw<float><<<grid_dim, block_dim, 0, stream>>>(d_q, d_k, d_v, d_out, d_l, d_m, batch, N, d);

    cudaMemcpy(out, d_out, batch * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(l, d_l, batch * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m, d_m, batch * N * sizeof(float), cudaMemcpyDeviceToHost);
    
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
    cudaFree(d_l);
    cudaFree(d_m);
    
}

}
}  
} 
