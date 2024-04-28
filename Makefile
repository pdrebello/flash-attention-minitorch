# For syntactic ease
#$(cp src/flash_attn_bw.cpp src/flash_attn_bw.cu)
#$(cp src/flash_attn_fw.cpp src/flash_attn_fw.cu)
#$(cp src/flash_attn2_fw.cpp src/flash_attn2_fw.cu)
#$(echo "HERE")

# Compiler and flags
CC = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS = -Xcompiler -fPIC

# Targets
all: combine layernorm softmax flash_fw flash_bw
flash: flash_fw flash_bw
flash_causal: flash_fw_causal flash_bw_causal
flash2: flash2_fw flash2_bw

combine: src/combine.cu
	$(NVCC) $(CUDAFLAGS) --shared src/combine.cu -o minitorch/cuda_kernels/combine.so

layernorm: src/layernorm_kernel.cu
	$(NVCC) $(CUDAFLAGS) --shared src/layernorm_kernel.cu -o minitorch/cuda_kernels/layernorm_kernel.so

softmax: src/softmax_kernel.cu
	$(NVCC) $(CUDAFLAGS) --shared src/softmax_kernel.cu -o minitorch/cuda_kernels/softmax_kernel.so

flash_fw: src/flash_attn_fw.cpp
	cp src/flash_attn_fw.cpp src/flash_attn_fw.cu
	$(NVCC) $(CUDAFLAGS) --shared src/flash_attn_fw.cu -o minitorch/cuda_kernels/flash_attn_fw.so

flash_bw: src/flash_attn_bw.cpp
	cp src/flash_attn_bw.cpp src/flash_attn_bw.cu 
	$(NVCC) $(CUDAFLAGS) --shared src/flash_attn_bw.cu -o minitorch/cuda_kernels/flash_attn_bw.so

flash_fw_causal: src/flash_attn_fw.cpp
	cp src/flash_attn_fw.cpp src/flash_attn_fw.cu
	$(NVCC) $(CUDAFLAGS) -DCAUSAL_BLOCKSPARSE --shared src/flash_attn_fw.cu -o minitorch/cuda_kernels/flash_attn_fw.so

flash_bw_causal: src/flash_attn_bw.cpp
	cp src/flash_attn_bw.cpp src/flash_attn_bw.cu 
	$(NVCC) $(CUDAFLAGS) -DCAUSAL_BLOCKSPARSE --shared src/flash_attn_bw.cu -o minitorch/cuda_kernels/flash_attn_bw.so

flash2_fw: src/flash_attn2_fw.cpp
	cp src/flash_attn2_fw.cpp src/flash_attn2_fw.cu
	$(NVCC) $(CUDAFLAGS) -DCAUSAL_BLOCKSPARSE --shared src/flash_attn2_fw.cu -o minitorch/cuda_kernels/flash_attn_fw.so

flash2_bw: src/flash_attn2_bw.cpp
	cp src/flash_attn2_bw.cpp src/flash_attn2_bw.cu
	$(NVCC) $(CUDAFLAGS) -DCAUSAL_BLOCKSPARSE --shared src/flash_attn2_bw.cu -o minitorch/cuda_kernels/flash_attn_bw.so

clean:
	rm minitorch/cuda_kernels/*.so
