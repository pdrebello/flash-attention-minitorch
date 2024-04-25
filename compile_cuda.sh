mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC

nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/flash_attn_fw.so --shared src/flash_attn_fw.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/flash_attn_bw.so --shared src/flash_attn_bw.cu -Xcompiler -fPIC
