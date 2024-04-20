#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cmath>
#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>
#define BASE_THREAD_NUM 32
#define TILE_SIZE 512
#define MBY4D 4

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

    int B_c = MBY4D; //BASE_THREAD_NUM; //on_chip_memory_size / (4 * d);  // Using 4 bytes per float
    int B_r = min(B_c, d); //min(on_chip_memory_size / (4 * d), d);
    int T_r = (N + B_r - 1)/ B_r;
    int T_c = (N +B_c -1)/ B_c;

    assert(TILE_SIZE > MBY4D * d);
    
    
    __shared__ float sram[TILE_SIZE * 10];
    float* Qi = sram;
    float* Kj = &sram[TILE_SIZE];
    float* Vj = &sram[TILE_SIZE * 2];
    float* Sij  = &sram[TILE_SIZE * 3];

    float* dKj = &sram[TILE_SIZE * 4];
    float* dVj = &sram[TILE_SIZE * 5];
    float* dPij  = &sram[TILE_SIZE * 6];
    float* Oi = &sram[TILE_SIZE * 7];
    float* dOi  = &sram[TILE_SIZE * 8];
    float* dSij  = &sram[TILE_SIZE * 9];
    
    __shared__ float lm_sram[MBY4D * 3];
    float* li = lm_sram;
    float* mi = &lm_sram[MBY4D];
    float* Di = &lm_sram[MBY4D * 2];
    

    int B_c_blocks = (B_c + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;
    int B_r_blocks = (B_r + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;
    int d_blocks = (d + BASE_THREAD_NUM - 1)/ BASE_THREAD_NUM;
    float threadSpecific_dQi[(TILE_SIZE /MBY4D + BASE_THREAD_NUM - 1)/BASE_THREAD_NUM];
    
    for(int j = 0; j < T_c; j++){
        // Loading
        for(int read_block = 0;read_block < B_c_blocks; read_block++){
            int tidx_ = read_block * BASE_THREAD_NUM + tidx;
            
            for(int y = 0; y < d; y++){
                if(tidx_ < B_c && j * B_c + tidx_ < N && tidx_y == 0){
                    Kj[tidx_ * d + y] = k[(j * B_c + tidx_) * d + y];
                    Vj[tidx_ * d + y] = v[(j * B_c + tidx_) * d + y];
                    dKj[tidx_ * d + y] = 0;
                    dVj[tidx_ * d + y] = 0;
                }
                else if(tidx_ < B_c && tidx_y == 0){
                    Kj[tidx_ * d + y] = 0;
                    Vj[tidx_ * d + y] = 0;
                    dKj[tidx_ * d + y] = 0;
                    dVj[tidx_ * d + y] = 0;
                }
            }
        }

        for(int i = 0; i < T_r; i++){
            // Loading 
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                
                if(tidx_ < B_r){
                    if(i * B_r + tidx_ < N){
                        li[tidx_] = l[i * B_r + tidx_];
                        mi[tidx_] = m[i * B_r + tidx_];
                        Di[tidx_] = 0;
                    }

                    for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){                   
                        int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;

                        if(tidx_y_ < d){ 
                            if(i * B_r + tidx_ < N){
                                Qi[tidx_ * d + tidx_y_]  = q[(i * B_r + tidx_) * d + tidx_y_];
                                threadSpecific_dQi[read_block_y]  = q_grad[(i * B_r + tidx_) * d + tidx_y_];
                                Oi[tidx_ * d + tidx_y_]  = out[(i * B_r + tidx_) * d + tidx_y_];
                                dOi[tidx_ * d + tidx_y_]  = out_grad[(i * B_r + tidx_) * d + tidx_y_];
        
                            }
                            else{
                                Qi[tidx_ * d + tidx_y_]  = 0;
                                threadSpecific_dQi[read_block_y]  = 0;
                                Oi[tidx_ * d + tidx_y_]  = 0;
                                dOi[tidx_ * d + tidx_y_]  = 0;   
                            }  
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
                        dSij[tidx_ * B_c + tidx_y_] = 0;
                        dPij[tidx_ * B_c + tidx_y_] = 0;
                    }   
                }
            }
            __syncthreads();
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        float S_acc = 0;
                        for(int y = 0; y < d; y++)
                            S_acc += (tau * Qi[tidx_ * d + y] * Kj[tidx_y_ * d + y]);
                        Sij[tidx_ * B_c + tidx_y_]  = S_acc; 
                    }
                }
            }
            __syncthreads();

            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        Sij[tidx_ * B_c + tidx_y_] = (1.0/li[tidx_]) * exp(Sij[tidx_ * B_c + tidx_y_] - mi[tidx_]);
                    } 
                }
            }
            __syncthreads();


            for(int read_block=0; read_block < B_c_blocks; read_block++){
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ <B_c && (j * B_c + tidx_ < N) && tidx_y_ < d){
                        float S_acc = dVj[tidx_ * d + tidx_y_];
                        for(int y = 0; y < B_r; y++)
                            S_acc += (Sij[y * B_c + tidx_] * dOi[y * d + tidx_y_]);  //(tau * Qi[tidx_ * d + y] * Kj[tidx_y_ * d + y]);
                        dVj[tidx_ * d + tidx_y_] = S_acc; 
                    }
                }
            }
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        float S_acc = 0;
                        for(int y = 0; y < d; y++){
                            if(tidx_y_ == 0)
                                Di[tidx_] += Oi[tidx_ * d + y] * dOi[tidx_ * d + y];
                            S_acc += (dOi[tidx_ * d + y] * Vj[tidx_y_ * d + y]);
                        }
                        dPij[tidx_ * B_c + tidx_y_] = S_acc; 
                    }
                }
            }
            __syncthreads();
            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < B_c_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < B_c && (j * B_c + tidx_y_ < N)){
                        dSij[tidx_ * B_c + tidx_y_] = Sij[tidx_ * B_c + tidx_y_] * (dPij[tidx_ * B_c + tidx_y_] - Di[tidx_]);
                    }
                }
            }
            __syncthreads();

            for(int read_block=0; read_block < B_r_blocks; read_block++){
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ <B_r && (i * B_r + tidx_ < N) && tidx_y_ < d){

                        float S_acc =  threadSpecific_dQi[read_block_y]; 
                        for(int y = 0; y < B_c; y++){
                            S_acc += (tau * dSij[tidx_ * B_c + y] * Kj[y * d + tidx_y_]); 
                        }
                        q_grad[i * d * B_r + tidx_ * d + tidx_y_] = S_acc;
                    }
                }
            }
            for(int read_block=0; read_block < B_c_blocks; read_block++){
                for(int read_block_y=0; read_block_y < d_blocks; read_block_y++){
                    int tidx_ = read_block * BASE_THREAD_NUM + tidx;
                    int tidx_y_ = read_block_y * BASE_THREAD_NUM + tidx_y;
                    
                    if(tidx_ <B_c && (j * B_c + tidx_ < N) && tidx_y_ < d){
                        float S_acc = dKj[tidx_ * d + tidx_y_];
                        for(int y = 0; y < B_r; y++){
                            S_acc += (tau * dSij[y * B_c + tidx_] * Qi[y * d + tidx_y_]); 
                        }
                        dKj[tidx_ * d + tidx_y_] = S_acc;
                    }
                }
            }
            __syncthreads();
        }

        for(int read_block = 0;read_block < B_c_blocks; read_block++){
            int tidx_ = read_block * BASE_THREAD_NUM + tidx;
            for(int y = 0; y < d; y++){
                if(tidx_ < B_c && j * B_c + tidx_ < N && tidx_y == 0){
                    k_grad[j * d * B_c + tidx_ * d + y] = dKj[tidx_ * d + y];
                    v_grad[j * d * B_c + tidx_ * d + y] = dVj[tidx_ * d + y];
                }
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
