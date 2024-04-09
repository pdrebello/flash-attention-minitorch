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
  batch_size, from_len = kt.bs_sl()
  _, to_len = kt.bs_sl(batch_size)
  nhead = kt.nhead

  batch_size, nhead, from_len, to_len = 64, 8, 512, 10
  
  print(
      "(batch_size, nhead, from_len, to_len),"
      f": ({batch_size}, {nhead}, {from_len}, {to_len})"
  )

  q = kt.rand((batch_size, nhead, from_len, to_len))
  k = kt.rand((batch_size, nhead, from_len, to_len))
  v = kt.rand((batch_size, nhead, from_len, to_len))

  def custom():
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

  def custom_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mtT = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)

    
    start_time = time.time()
    datatype = np.float32
    mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, from_len, from_len), dtype=datatype), 1) 
    mask = mask * 0
    M = minitorch.tensor_from_numpy(mask, backend=backend)
      
    cust_out = minitorch.nn.softmax(((q_mt @ k_mtT)/np.sqrt(to_len)) +M, dim=3) @ v_mt
    end_time = time.time()

    cust_out = torch.tensor(cust_out._tensor._storage).float().cuda()
    return [
        cust_out,
    ], end_time - start_time

  def baseline_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    
    #k_mt = k_mt.view(0, 1, 3, 2)
    start_time = time.time()
    res = minitorch.nn.softmax(((q_mt @ k_mt)/np.sqrt(to_len)) , dim=3) @ v_mt
    end_time = time.time()

    res = torch.tensor(res._tensor._storage).float().cuda()
    return kt.norm_res_list(res), end_time - start_time
      
  def baseline():
    cust_out = torch.zeros_like(q)
    start_time = time.time()

    for batch_idx in range(batch_size):
        for head in range(nhead):
            cust_out[batch_idx, head, :, :],_,_ = flash_attention(q[batch_idx, head, :, :].cpu(), k[batch_idx, head, :, :].cpu(), v[batch_idx, head, :, :].cpu())
    end_time = time.time()

    return [
        cust_out,
    ], end_time - start_time
  return custom, custom_minitorch #baseline


kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_flash_attn_fw'
)