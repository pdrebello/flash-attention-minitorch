import numpy as np
import time

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

from flash_attn_python import flash_attention 

@kt.case(atol=1e-3, rtol=1e-3, ntest=5)
def test_launch_flash_attn_fw():
  #batch_size, from_len = kt.bs_sl()
  #_, to_len = kt.bs_sl(batch_size)
  #nhead = kt.nhead

  #batch_size, nhead, from_len, to_len = 1, 1, 125, 13
  
  print(
      "(batch_size, nhead, from_len, to_len),"
      f": ({batch_size}, {nhead}, {from_len}, {to_len})"
  )

  q = kt.rand((batch_size, nhead, from_len, to_len))
  k = kt.rand((batch_size, nhead, from_len, to_len))
  v = kt.rand((batch_size, nhead, from_len, to_len))

  def flash_attention_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)

    
    #cust_out = torch.zeros_like(q)
    start_time = time.time()
    out_mt = q_mt.flash_attn(k_mt, v_mt)
    end_time = time.time()

    cust_out = torch.tensor(out_mt._tensor._storage).float().cuda()
    return [
        cust_out,
    ], end_time - start_time

  #### TORCH BACKPROP #####
  def attention_torch():
    
    start_time = time.time()
    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            temp_q = q[batch_idx, head_idx, :, :].clone()
            temp_q.requires_grad = True
            temp_k = k[batch_idx, head_idx, :, :].clone()
            temp_k.requires_grad = True
            temp_v = v[batch_idx, head_idx, :, :].clone()
            temp_v.requires_grad = True
            out[batch_idx, head_idx] = compute_attention(temp_q, temp_k, temp_v)

    end_time = time.time()

    return [
        out
    ], end_time - start_time

  def attention_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    
      
    #k_mt = k_mt.view(0, 1, 3, 2)
    start_time = time.time()
    drop = minitorch.tensor_from_numpy(np.ones((from_len, from_len),dtype=float), backend=backend)
    mask = minitorch.tensor_from_numpy(np.zeros((from_len, from_len),dtype=float), backend=backend)
    res = (q_mt @ k_mt)
    res /= np.sqrt(to_len)
    res += mask
    res = minitorch.nn.softmax(res , dim=3)
    res = drop * res
    res = res @ v_mt
    #res = (drop * minitorch.nn.softmax(() + mask , dim=3)) @ v_mt
    end_time = time.time()

    res = torch.tensor(res._tensor._storage).float().cuda()
    return kt.norm_res_list(res), end_time - start_time
      
  def flash_attention_torch():
    cust_out = torch.zeros_like(q)
    start_time = time.time()

    for batch_idx in range(batch_size):
        for head in range(nhead):
            cust_out[batch_idx, head, :, :],_,_ = flash_attention(q[batch_idx, head, :, :].cpu(), k[batch_idx, head, :, :].cpu(), v[batch_idx, head, :, :].cpu())
    end_time = time.time()

    return [
        cust_out,
    ], end_time - start_time
  return flash_attention_minitorch, attention_minitorch #flash_attention_torch #


kt.init(device="cuda:0", nhead=8)
#kt.run(
#  'test_launch_flash_attn_fw'
#)

#for batch_size in [128]:
#    for nhead in [8]:
#        for from_len in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#            for to_len  in [1, 2, 4, 8, 15]:
#                kt.run('test_launch_flash_attn_fw')

#for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
#    for nhead in [1]:
#        for from_len in [40]:
#            for to_len  in [15]:
#                kt.run('test_launch_flash_attn_fw')
for batch_size in [1]:
    for nhead in [128]:
        #for from_len in [3, 40, 90]:
        #    for to_len  in [3, 17, 256]:
        for from_len in [512]:
            for to_len  in [128]:
                #assert(to_len * 4 <= 1024)
                kt.run('test_launch_flash_attn_fw')

#batch_size, nhead, from_len, to_len = 2, 2, 489, 13
#kt.run('test_launch_flash_attn_fw')
"""
for batch_size in [1, 31, 127, 511]:
    for nhead in [1, 7]:
        for from_len in [31, 124, 489]:
            for to_len  in [4, 13]:
                #batch_size, nhead, from_len, to_len = 6, 8, 31, 18
                kt.run('test_launch_flash_attn_fw')
"""


