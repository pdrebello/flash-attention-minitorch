from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import minitorch

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
lib_flash_attn_fw = ctypes.CDLL("minitorch/cuda_kernels/flash_attn_fw.so")
lib_flash_attn_bw = ctypes.CDLL("minitorch/cuda_kernels/flash_attn_bw.so")
datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_double,                                                         # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
      batch_size, nhead, from_len, to_len = inp.shape
      is_dec_self_attn = False
      stream = torch.cuda.current_stream().cuda_stream

      lib_softmax.launch_attn_softmax.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_void_p
      ]
      lib_softmax.launch_attn_softmax.restype = None

      lib_softmax.launch_attn_softmax(
        inp._tensor._storage,
        mask._tensor._storage,
        batch_size,
        nhead,
        from_len,
        to_len,
        is_dec_self_attn,
        stream
      ) 

      return inp

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
      #   BEGIN ASSIGN3_1
      batch_size, nhead, from_len, to_len = soft_inp.shape
      assert(out_grad.shape == soft_inp.shape)
      rows = batch_size * nhead * from_len
      softmax_len = to_len
      
      stream = torch.cuda.current_stream().cuda_stream

      lib_softmax.launch_attn_softmax_bw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_softmax.launch_attn_softmax_bw.restype = None

      lib_softmax.launch_attn_softmax_bw(
        out_grad._tensor._storage,
        soft_inp._tensor._storage,
        rows,
        softmax_len,
        stream
      ) 
      return out_grad
      #   END ASSIGN3_1

    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
      #   BEGIN ASSIGN3_2
      batch_size, hidden_dim = inp.shape
      ln_res = inp.zeros((batch_size, hidden_dim))
      vars = minitorch.zeros((batch_size,))
      means = minitorch.zeros((batch_size,))
      stream = torch.cuda.current_stream().cuda_stream
        
      lib_layernorm.launch_layernorm.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_layernorm.launch_layernorm.restype = None

      lib_layernorm.launch_layernorm(
        ln_res._tensor._storage,
        vars._tensor._storage,
        means._tensor._storage,
        inp._tensor._storage,
        gamma._tensor._storage,
        beta._tensor._storage,
        batch_size,
        hidden_dim,
        stream
      ) 
      return ln_res, vars, means
      #   END ASSIGN3_2
      
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, var: Tensor, mean: Tensor):
      #   BEGIN ASSIGN3_2
        batch_size, hidden_dim = inp.shape
        gamma_grad = inp.zeros(gamma.shape)
        betta_grad = inp.zeros(beta.shape)
        inp_grad = inp.zeros(inp.shape)
        
        stream1 = torch.cuda.current_stream().cuda_stream
        stream2 = torch.cuda.current_stream().cuda_stream
        
        lib_layernorm.launch_layernorm_bw.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p
        ]
        
        lib_layernorm.launch_layernorm_bw.restype = None
        
        lib_layernorm.launch_layernorm_bw(
            gamma_grad._tensor._storage,
            betta_grad._tensor._storage,
            inp_grad._tensor._storage,
            out_grad._tensor._storage,
            inp._tensor._storage,
            gamma._tensor._storage,
            beta._tensor._storage,
            var._tensor._storage,
            mean._tensor._storage,
            batch_size,
            hidden_dim,
            stream1,
            stream2
        )

        return inp_grad, gamma_grad, betta_grad
      #   END ASSIGN3_2
      
    @staticmethod
    def flash_attn_fw(q: Tensor, k: Tensor, v: Tensor):
      batch_size, nhead, from_len, to_len = q.shape
      assert(q.shape == k.shape)
      assert(q.shape == v.shape)
      assert((q._tensor._strides - k._tensor._strides).sum() ==0)
      assert((q._tensor._strides - v._tensor._strides).sum() ==0)
      stream = torch.cuda.current_stream().cuda_stream
        
      out = q.zeros(q.shape)
      l = k.zeros((batch_size, nhead, from_len))
      m = v.zeros((batch_size, nhead, from_len))  -1000000
        
      q = q.contiguous().view(np.prod(q.shape[:-2]), q.shape[-2], q.shape[-1])
      k = k.contiguous().view(np.prod(k.shape[:-2]), k.shape[-2], k.shape[-1])
      v = v.contiguous().view(np.prod(v.shape[:-2]), v.shape[-2], v.shape[-1])
      out = out.contiguous().view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
      l = l.contiguous().view(np.prod(l.shape[:-2]), l.shape[-2], l.shape[-1])
      m = m.contiguous().view(np.prod(m.shape[:-2]), m.shape[-2], m.shape[-1])
        
      lib_flash_attn_fw.launch_flash_attn_fw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # q
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # k
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # v
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # l
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # m
        ctypes.c_int, # batch_size
        ctypes.c_int, # from_len
        ctypes.c_int, # to_len
        ctypes.c_void_p
      ]
      lib_flash_attn_fw.launch_flash_attn_fw.restype = None

      lib_flash_attn_fw.launch_flash_attn_fw(
        q._tensor._storage,
        k._tensor._storage,
        v._tensor._storage,
        out._tensor._storage,
        l._tensor._storage,
        m._tensor._storage,
        batch_size * nhead,
        from_len,
        to_len,
        stream
      ) 
      out = out.view(batch_size, nhead, from_len, to_len) 
      l = l.view(batch_size, nhead, from_len) 
      m = m.view(batch_size, nhead, from_len) 
      
      return out, l, m

    @staticmethod
    def flash_attn_bw(q: Tensor, k: Tensor, v: Tensor, out: Tensor, out_grad: Tensor, l: Tensor, m: Tensor):
      #   BEGIN ASSIGN3_1
      """
      print("K")
      print(k.to_numpy()[0,0])
      print("V")
      print(v.to_numpy()[0,0])
      print("Q")
      print(q.to_numpy()[0,0])
      print("Out")
      print(out.to_numpy()[0,0])
      print("Out Grad")
      print(out_grad.to_numpy()[0,0])
      
      print("l")
      print(l.to_numpy())
      print("m")
      print(m.to_numpy())
        """
        
      batch_size, nhead, from_len, to_len = q.shape
      assert(q.shape == k.shape)
      assert(q.shape == v.shape)
      assert(q.shape == out.shape)
      assert(q.shape == out_grad.shape)
      assert(l.shape == (batch_size, nhead, from_len))
      assert(m.shape == (batch_size, nhead, from_len))
      assert((q._tensor._strides - k._tensor._strides).sum() ==0)
      assert((q._tensor._strides - v._tensor._strides).sum() ==0)
      assert((q._tensor._strides - out._tensor._strides).sum() ==0)
      assert((q._tensor._strides - out_grad._tensor._strides).sum() ==0)
      stream = torch.cuda.current_stream().cuda_stream

      q = q.contiguous().view(np.prod(q.shape[:-2]), q.shape[-2], q.shape[-1])
      k = k.contiguous().view(np.prod(k.shape[:-2]), k.shape[-2], k.shape[-1])
      v = v.contiguous().view(np.prod(v.shape[:-2]), v.shape[-2], v.shape[-1])
      out = out.contiguous().view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
      out_grad = out_grad.contiguous().view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])

      l = l.contiguous().view(np.prod(l.shape[:-2]), l.shape[-2], l.shape[-1])
      m = m.contiguous().view(np.prod(m.shape[:-2]), m.shape[-2], m.shape[-1])
      
      q_grad = q.zeros(q.shape)
      k_grad = k.zeros(k.shape)
      v_grad = v.zeros(v.shape)
        
      lib_flash_attn_bw.launch_flash_attn_bw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # q
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # k
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # v
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # out_grad
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # q_grad
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # k_grad
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # v_grad
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # l
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'), # m
        ctypes.c_int, # batch_size
        ctypes.c_int, # from_len
        ctypes.c_int, # to_len
        ctypes.c_void_p
      ]
      lib_flash_attn_bw.launch_flash_attn_bw.restype = None

      lib_flash_attn_bw.launch_flash_attn_bw(
        q._tensor._storage,
        k._tensor._storage,
        v._tensor._storage,
        out._tensor._storage,
        out_grad._tensor._storage,
        q_grad._tensor._storage,
        k_grad._tensor._storage,
        v_grad._tensor._storage,
        l._tensor._storage,
        m._tensor._storage,
        batch_size * nhead,
        from_len,
        to_len,
        stream
      ) 
      q_grad = q_grad.view(batch_size, nhead, from_len, to_len) 
      k_grad = k_grad.view(batch_size, nhead, from_len, to_len) 
      v_grad = v_grad.view(batch_size, nhead, from_len, to_len) 
      
      return q_grad, k_grad, v_grad


