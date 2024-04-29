#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cmath>
#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include <sys/time.h>


#include <cooperative_groups.h>
#define BASE_THREAD_NUM 32
#define TILE_SIZE 2024
#define MBY4D 16

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {

template <typename T> //, int block_dim, int ele_per_thread>
__global__ void flash_attn_fw(T *q, T *k, T *v, T *out, T *l, T *m, int batch, int N, int d, bool causal_mask=false) {
    
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
    
    float tau = sqrt(1.0/d);    
    int B_c = MBY4D; //BASE_THREAD_NUM; //on_chip_memory_size / (4 * d);  // Using 4 bytes per float
    int B_r = min(B_c, d); //min(on_chip_memory_size / (4 * d), d);
    int T_r = (N + B_r - 1)/ B_r;
    int T_c = (N +B_c -1)/ B_c;

    assert(TILE_SIZE > MBY4D * d);
    //assert(d < TILE_SIZE/BASE_THREAD_NUM);
    __shared__ float sram[TILE_SIZE * 6];
    float* Qi = sram;
    float* Kj = &sram[TILE_SIZE];
    float* Vj = &sram[TILE_SIZE * 2];
    float* Sij  = &sram[TILE_SIZE * 3];
    
    float* tempPRO  = &sram[TILE_SIZE * 4];
    float* Oi  = &sram[TILE_SIZE * 5];

    __shared__ float lm_sram[MBY4D * 3];
    float* li = lm_sram;
    float* mi = &lm_sram[MBY4D];
    float* mi_prev = &lm_sram[MBY4D*2];

    int B_c_blocks = (B_c + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;
    int B_r_blocks = (B_r + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;
    int d_blocks = (d + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;

    
#ifdef TIME
    clock_t start_time, end_time;
#endif    
    int i = blockIdx.y;
    //for(int i = 0; i < T_r; i++){
        // Loading
        for(int read_block=0; read_block < B_r_blocks; read_block++){
            int tidx_ = read_block * BASE_THREAD_NUM + tidx;
            if(tidx_ < B_r){
                li[tidx_] = 0;
                mi[tidx_] = -10000000;
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){                   
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_y_ < d){ 
                        if(i * B_r + tidx_ < N){
                            Qi[tidx_ * d + tidx_y_]  = q[(i * B_r + tidx_) * d + tidx_y_];
                            Oi[tidx_ * d + tidx_y_]  = 0;
                        }
                        else{
                            Qi[tidx_ * d + tidx_y_]  = 0;
                            Oi[tidx_ * d + tidx_y_]  = 0;
                        }
                    }
                }
            }
        }

        __syncthreads();
        
        for(int j = 0; j < T_c; j++){
            // Loading 
#ifdef CAUSAL_BLOCKSPARSE
            if(causal_mask && (j * B_c > (i+1) * B_r -1)){
                continue;
            }
#endif
#ifdef TIME
            if(tidx==0 and tidx_y == 0)
                 start_time  = clock();
#endif

            for(int read_block = 0;read_block < B_c_blocks; read_block++){
                for(int read_block_y = 0; read_block_y < d_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ < B_c && j * B_c + tidx_ < N && tidx_y_ < d){
                        Kj[tidx_ * d + tidx_y_] = k[(j * B_c + tidx_) * d + tidx_y_];
                        Vj[tidx_ * d + tidx_y_] = v[(j * B_c + tidx_) * d + tidx_y_];
                    }
                    else if(tidx_ < B_c && tidx_y == 0){
                        Kj[tidx_ * d + tidx_y_] = 0;
                        Vj[tidx_ * d + tidx_y_] = 0;
                    }
                }
            }
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                if(tidx_ < B_r){
                    for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){                   
                        int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                        if(tidx_y_ < d){ 
                            tempPRO[tidx_ * d + tidx_y_] = 0; 
                        }
                    }
                }
            }
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_ <B_r &&  tidx_y_ < B_c ){
                        Sij[tidx_ * B_c + tidx_y_] = 0;
                    }   

                }
            }
            

            __syncthreads();
#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: init\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif
            
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        if(!causal_mask || j * B_c + tidx_y_ <= i * B_r + tidx_){
                            float S_acc = 0;
                            for(int y = 0; y < d; y++)
                                S_acc += (tau * Qi[tidx_ * d + y] * Kj[tidx_y_ * d + y]);
                            Sij[tidx_ * B_c + tidx_y_]  = S_acc; 
                        }
                        else
                            Sij[tidx_ * B_c + tidx_y_]  = -10000000;
                    }
                }
            }
            __syncthreads();

#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: S\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif

            for(int read_block=0; read_block < B_r_blocks; read_block++){
                int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                if(tidx_ < B_r && (i * B_r + tidx_ < N)  && tidx_y == 0){
                    mi_prev[tidx_] = mi[tidx_];
                }
                for(int y = 0; y < B_c; y++){
                    if(tidx_ < B_r && (i * B_r + tidx_ < N) && tidx_y == 0 && (j * B_c + y < N)){
                        mi[tidx_] = max(mi[tidx_], Sij[tidx_ * B_c + y]);
                    }  
                }
                if(tidx_ < B_r && (i * B_r + tidx_ < N)  && tidx_y == 0){
                    li[tidx_] = exp(mi_prev[tidx_] - mi[tidx_]) * li[tidx_];
                }
            }
            __syncthreads();

#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: Mij\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        Sij[tidx_ * B_c + tidx_y_] = exp(Sij[tidx_ * B_c + tidx_y_] - mi[tidx_]);
                    } 
                }
            }
            __syncthreads();
#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: Pij\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                for(int y = 0; y < B_c; y++){
                    if(tidx_ < B_r && (i * B_r + tidx_ < N) && tidx_y == 0 && (j * B_c + y < N)){
                        li[tidx_] += Sij[tidx_ * B_c + y];
                    }  
                }
            }
            __syncthreads();

#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: lij\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif

            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    for(int y = 0; y < B_c; y++){
                        if(tidx_ < B_r && (i * B_r + tidx_ < N) && tidx_y_ < d){
                            tempPRO[tidx_ * d + tidx_y_] += (Sij[tidx_ * B_c + y] * Vj[y * d + tidx_y_]);
                        }   
                    }
                }
            }
            __syncthreads();
#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: temp\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                if(tidx_ < B_r && i * B_r + tidx_ < N){
                    for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){                   
                        int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                        if(tidx_y_ < d){
                            if(j == 0)
                                Oi[tidx_ * d + tidx_y_] = tempPRO[tidx_ * d + tidx_y_];
                            else
                                Oi[tidx_ * d + tidx_y_] = exp(mi_prev[tidx_] - mi[tidx_]) * Oi[tidx_ * d + tidx_y_] + tempPRO[tidx_ * d + tidx_y_];
                        }
                    }
                }
            }
            __syncthreads();
#ifdef TIME
            if(tidx==0 and tidx_y == 0){
                end_time  = clock();
                printf("%f: Output\n", 1000000.0*(end_time - start_time));
                start_time = clock();
            }
#endif
        }
        __syncthreads();
        for(int read_block=0; read_block < B_r_blocks; read_block++){
            int tidx_ = read_block * BASE_THREAD_NUM + tidx;
            if(tidx_ < B_r){
                if(i * B_r + tidx_ < N && tidx_y == 0){
                    l[i * B_r + tidx_] = mi[tidx_] + log(li[tidx_]);
                }
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){                   
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_y_ < d){ 
                        if(i * B_r + tidx_ < N){
                            out[(i * B_r + tidx_) * d + tidx_y_] = (1.0/li[tidx_]) * Oi[tidx_ * d + tidx_y_];
                        }
                    }
                }
            }
        }
        __syncthreads();
    //}
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
    bool causal_mask,
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

    int B_r = min(MBY4D, d); 
    int T_r = (N + B_r - 1)/ B_r;
    
    dim3 grid_dim(batch, T_r);  // batch_size x num_heads
    dim3 block_dim(BASE_THREAD_NUM, BASE_THREAD_NUM);

    flash_attn_fw<float><<<grid_dim, block_dim, 0, stream>>>(d_q, d_k, d_v, d_out, d_l, d_m, batch, N, d, causal_mask);

    cudaMemcpy(out, d_out, batch * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(l, d_l, batch * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m, d_m, batch * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    
    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        gpuErrchk(err);
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
