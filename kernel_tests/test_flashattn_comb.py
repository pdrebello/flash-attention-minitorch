import numpy as np
import time

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

from flash_attn_python import flash_attention , flash_attention_backward, compute_attention

@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_flash_attn_comb():
  #batch_size, from_len = kt.bs_sl()
  #_, to_len = kt.bs_sl(batch_size)
  #nhead = kt.nhead

  #batch_size, nhead, from_len, to_len = 1, 1, 125, 13
  
  print(
      "(batch_size, nhead, from_len, to_len),"
      f": ({batch_size}, {nhead}, {from_len}, {to_len})"
  )

  out_grad = kt.ones((batch_size, nhead, from_len, to_len))
  q = kt.rand((batch_size, nhead, from_len, to_len))
  k = kt.rand((batch_size, nhead, from_len, to_len))
  v = kt.rand((batch_size, nhead, from_len, to_len))

  ### OUR SOLUTION ####
  def flash_attention_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    
    start_time = time.time()
    out_mt = q_mt.flash_attn(k_mt, v_mt)
    out_mt.backward(out_grad_mt)
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()

    return [
        dq,dk,dv
    ], end_time - start_time

  #### MINITORCH BACKWARD #####
  def attention_minitorch():
    start_time = time.time()
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    result = minitorch.softmax(((q_mt @ k_mt)/np.sqrt(to_len)) , dim=3) @ v_mt
    result = result.sum()
    result.backward()
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().transpose(0,1,3,2), dtype=torch.float32).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()

    
    return [dq, dk,dv], end_time - start_time
  return flash_attention_minitorch, attention_minitorch, 


kt.init(device="cuda:0", nhead=8)
#for batch_size in [128]:
#    for nhead in [8]:
#        for from_len in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#            for to_len  in [1, 2, 4, 8, 15]:
#                kt.run('test_launch_flash_attn_comb')
#for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
#    for nhead in [1]:
#        for from_len in [40]:
#            for to_len  in [15]:
#                kt.run('test_launch_flash_attn_comb')
for batch_size in [128]:
    for nhead in [8]:
        for from_len in [40]:
            for to_len  in [32]:
                kt.run('test_launch_flash_attn_comb')
#kt.run(
#  'test_launch_flash_attn_bw'
#)