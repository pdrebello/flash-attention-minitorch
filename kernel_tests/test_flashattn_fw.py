import numpy as np
import time
import argparse

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

from flash_attn_python import flash_attention, flash_attention2

datatype = np.float32

def create_causal_mask(bs, nh, seq_len, seq_dim, backend):
    mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1) 
    return minitorch.tensor_from_numpy(mask, backend=backend)


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
    
    start_time = time.time()    
    drop = minitorch.tensor_from_numpy(np.ones((from_len, from_len), dtype=datatype), backend=backend)
    res = (q_mt @ k_mt)/np.sqrt(to_len)
    
    mask_mt = create_causal_mask(batch_size, nhead, from_len, to_len, q_mt.backend)
    res += mask_mt
    res = (drop * minitorch.nn.softmax(res , dim=3)) @ v_mt
    #res = (drop * minitorch.nn.softmax(() + mask , dim=3)) @ v_mt
    end_time = time.time()

    res = torch.tensor(res._tensor._storage.astype(datatype)).cuda()
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

  def flash_attention2_torch():
    cust_out = torch.zeros_like(q)
    start_time = time.time()

    for batch_idx in range(batch_size):
        for head in range(nhead):
            cust_out[batch_idx, head, :, :],_ = flash_attention2(q[batch_idx, head, :, :].cpu(), k[batch_idx, head, :, :].cpu(), v[batch_idx, head, :, :].cpu())
    end_time = time.time()

    return [
        cust_out,
    ], end_time - start_time


  def flash_attention_minitorch(causal=False, flash2=False):
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    start_time = time.time()
    
    if(flash2):
        out_mt = q_mt.flash_attn2(k_mt, v_mt, causal_mask=True) 
    elif(causal):
        out_mt = q_mt.flash_attn_causal(k_mt, v_mt, causal_mask=True) 
    else:
        out_mt = q_mt.flash_attn(k_mt, v_mt, causal_mask=True)
    end_time = time.time()

    cust_out = torch.tensor(out_mt._tensor._storage.astype(datatype)).cuda()
    return [cust_out], end_time - start_time

  def flash_attention_causal_minitorch():
      return flash_attention_minitorch(causal=True)
      
  def flash_attention2_minitorch():
      return flash_attention_minitorch(flash2 = True)

  if(args.flash2): 
      return flash_attention2_minitorch, attention_minitorch 
  elif(args.causal):
      print(args.causal)
      return flash_attention_causal_minitorch, attention_minitorch 
  else:
      return flash_attention_minitorch, attention_minitorch       


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='mask, dropout, flash attention 2')
    parser.add_argument('--flash2', action='store_true')
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--N', type=int)
    args = parser.parse_args()
    
    kt.init(device="cuda:0", nhead=8)
    #batch_size = int(8192/args.N)
    batch_size = 1
    for batch_size in [batch_size]:
        for nhead in [8]:
            for from_len in [args.N]: 
                for to_len  in [64]:
                    try:
                        kt.run('test_launch_flash_attn_fw')
                        print("", end="", flush=True)
                    except Exception as e:
                        print(e)
                        pass
    

