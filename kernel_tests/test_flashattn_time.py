import numpy as np
import time
import argparse

import torch

from test_utils_timing import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

from flash_attn_python import flash_attention , flash_attention_backward, flash_attention2 , flash_attention2_backward, compute_attention
from test_flashattn_fw import create_causal_mask

datatype = np.float32

@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_flash_attn_time():
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
      
  #### MINITORCH BACKWARD #####
  def attention_minitorch():
    start_time = time.time()
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    drop = minitorch.tensor_from_numpy(np.ones((from_len, from_len),dtype=datatype), backend=backend)
    res = (q_mt @ k_mt)/np.sqrt(to_len)    
    mask_mt = create_causal_mask(batch_size, nhead, from_len, to_len, q_mt.backend)
    res += mask_mt
    res = minitorch.nn.softmax(res , dim=3)
    res = (drop * res)
    res = res  @ v_mt
    res = res.sum()
    inter_time = time.time()
    res.backward()
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().transpose(0,1,3,2).astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()
    
    return [dq, dk,dv], [end_time - start_time, inter_time - start_time, end_time - inter_time]

  ### OUR SOLUTION ####
  def flash_attention_minitorch(causal=False, flash2=False):
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    if(flash2):
        out_mt = q_mt.flash_attn2(k_mt, v_mt, causal_mask=True) 
    elif(causal):
        out_mt = q_mt.flash_attn_causal(k_mt, v_mt, causal_mask=True) 
    else:
        out_mt = q_mt.flash_attn(k_mt, v_mt, causal_mask=True)
    inter_time = time.time()
    out_mt.backward(out_grad_mt)
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()

    return [ dq,dk,dv], [end_time - start_time, inter_time - start_time, end_time - inter_time]
      
  def flash_attention_causal_minitorch():
      return flash_attention_minitorch(causal=True)
      
  def flash_attention2_minitorch():
      return flash_attention_minitorch(flash2 = True)

      
  return [flash_attention_minitorch, flash_attention_causal_minitorch, flash_attention2_minitorch], attention_minitorch 


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='mask, dropout, flash attention 2')
    parser.add_argument('--N', type=int)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    kt.init(device="cuda:0", nhead=8)
    #batch_size = int(8192/args.N)
    batch_size = 8
    #batch_size = int(256/args.N)
    
    for batch_size in [args.batch_size]:
        for nhead in [args.heads]:
            for from_len in [args.N]: 
                for to_len  in [args.d]:
                    try:
                        kt.run('test_launch_flash_attn_time')
                        print("", end="", flush=True)
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        pass
