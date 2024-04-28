import numpy as np
import time
import argparse

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

from flash_attn_python import flash_attention , flash_attention_backward, flash_attention2 , flash_attention2_backward, compute_attention
from test_flashattn_fw import create_causal_mask

datatype = np.float32

@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_flash_attn_bw():
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

  #### TORCH BACKPROP #####
  def attention_torch():
    out = kt.zeros((batch_size, nhead, from_len, to_len))
    l = kt.zeros((batch_size, nhead, from_len))
    m = kt.zeros((batch_size, nhead, from_len))

    dq = kt.rand((batch_size, nhead, from_len, to_len))
    dk = kt.rand((batch_size, nhead, from_len, to_len))
    dv = kt.rand((batch_size, nhead, from_len, to_len))

    start_time = time.time()
    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            temp_q = q[batch_idx, head_idx, :, :].clone()
            temp_q.requires_grad = True
            temp_k = k[batch_idx, head_idx, :, :].clone()
            temp_k.requires_grad = True
            temp_v = v[batch_idx, head_idx, :, :].clone()
            temp_v.requires_grad = True
            temp_output = compute_attention(temp_q, temp_k, temp_v)
            temp_output = temp_output.sum()
            temp_output.backward()
            dq[batch_idx, head_idx, :, :] = temp_q.grad
            dk[batch_idx, head_idx, :, :] = temp_k.grad
            dv[batch_idx, head_idx, :, :] = temp_v.grad

    end_time = time.time()

    return [
        dq,dk,dv
    ], end_time - start_time

  #### PYTHON FLASH ATTENTION BACKWARD FROM NOTEBOOK ####
  def flash_attention_torch():
    start_time = time.time()
      
    out = kt.zeros((batch_size, nhead, from_len, to_len))
    l = kt.zeros((batch_size, nhead, from_len))
    m = kt.zeros((batch_size, nhead, from_len))

    dq = kt.rand((batch_size, nhead, from_len, to_len))
    dk = kt.rand((batch_size, nhead, from_len, to_len))
    dv = kt.rand((batch_size, nhead, from_len, to_len))
    start_time = time.time()
    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            out[batch_idx, head_idx, :, :],l[batch_idx, head_idx, :],m[batch_idx, head_idx, :] = flash_attention(q[batch_idx, head_idx, :, :], \
                                                     k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :])

    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            dq[batch_idx, head_idx, :, :], dk[batch_idx, head_idx, :, :], dv[batch_idx, head_idx, :, :] = flash_attention_backward(q[batch_idx, head_idx, :, :],\
                                                  k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :], \
                                                  out[batch_idx, head_idx, :, :], out_grad[batch_idx, head_idx, :, :],
                                                  l[batch_idx, head_idx, :], m[batch_idx, head_idx, :])
    end_time = time.time()
      #kt.norm_res_list(
    return [dq, dk,dv], end_time - start_time

  #### PYTHON FLASH ATTENTION BACKWARD FROM NOTEBOOK ####
  def flash_attention2_torch():
    start_time = time.time()
      
    out = kt.zeros((batch_size, nhead, from_len, to_len))
    l = kt.zeros((batch_size, nhead, from_len))
    m = kt.zeros((batch_size, nhead, from_len))

    dq = kt.rand((batch_size, nhead, from_len, to_len))
    dk = kt.rand((batch_size, nhead, from_len, to_len))
    dv = kt.rand((batch_size, nhead, from_len, to_len))
    start_time = time.time()
    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            out[batch_idx, head_idx, :, :],l[batch_idx, head_idx, :] = flash_attention2(q[batch_idx, head_idx, :, :], \
                                                     k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :])

    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            dq[batch_idx, head_idx, :, :], dk[batch_idx, head_idx, :, :], dv[batch_idx, head_idx, :, :] = flash_attention2_backward(q[batch_idx, head_idx, :, :],\
                                                  k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :], \
                                                  out[batch_idx, head_idx, :, :], out_grad[batch_idx, head_idx, :, :],
                                                  l[batch_idx, head_idx, :])
    end_time = time.time()
      #kt.norm_res_list(
    return [dq, dk,dv], end_time - start_time

  #### PYTHON FLASH ATTENTION BACKWARD FROM NOTEBOOK ####
  def flash_attention2_torch():
    start_time = time.time()
      
    out = kt.zeros((batch_size, nhead, from_len, to_len))
    L = kt.zeros((batch_size, nhead, from_len))
    

    dq = kt.rand((batch_size, nhead, from_len, to_len))
    dk = kt.rand((batch_size, nhead, from_len, to_len))
    dv = kt.rand((batch_size, nhead, from_len, to_len))
    start_time = time.time()

    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            out[batch_idx, head_idx, :, :],L[batch_idx, head_idx, :] = flash_attention2(q[batch_idx, head_idx, :, :], \
                                                     k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :])

    for batch_idx in range(batch_size):
        for head_idx in range(nhead):
            dq[batch_idx, head_idx, :, :], dk[batch_idx, head_idx, :, :], dv[batch_idx, head_idx, :, :] = flash_attention2_backward(q[batch_idx, head_idx, :, :],\
                                                  k[batch_idx, head_idx, :, :], v[batch_idx, head_idx, :, :], \
                                                  out[batch_idx, head_idx, :, :], out_grad[batch_idx, head_idx, :, :],
                                                  L[batch_idx, head_idx, :])
    end_time = time.time()
      #kt.norm_res_list(
    return [dq, dk,dv], end_time - start_time
      
  #### MINITORCH BACKWARD #####
  def attention_minitorch():
    start_time = time.time()
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.permute(0,1,3,2).clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    drop = minitorch.tensor_from_numpy(np.ones((from_len, from_len),dtype=datatype), backend=backend)
    res = (q_mt @ k_mt)/np.sqrt(to_len)
    mask_mt = create_causal_mask(batch_size, nhead, from_len, to_len, q_mt.backend)
    res += mask_mt
    result = (drop * minitorch.nn.softmax(res , dim=3)) @ v_mt

    start_time = time.time()
    result = result.sum()
    result.backward()
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().transpose(0,1,3,2).astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()

    
    return [dq, dk,dv], end_time - start_time

  ### OUR SOLUTION ####
  def flash_attention_minitorch(causal=False, flash2=False):
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    if(flash2):
        out_mt = q_mt.flash_attn2(k_mt, v_mt, causal_mask=True) 
    elif(causal):
        out_mt = q_mt.flash_attn_causal(k_mt, v_mt, causal_mask=True) 
    else:
        out_mt = q_mt.flash_attn(k_mt, v_mt, causal_mask=True)

    start_time = time.time()
    out_mt.backward(out_grad_mt)
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()

    return [ dq,dk,dv], end_time - start_time
      
  def flash_attention_causal_minitorch():
      return flash_attention_minitorch(causal=True)
      
  def flash_attention2_minitorch():
      return flash_attention_minitorch(flash2 = True)

  if(args.flash2): 
      return flash_attention2_minitorch, attention_minitorch 
  elif(args.causal):
      return flash_attention_causal_minitorch, attention_minitorch 
  else:
      return flash_attention_minitorch, attention_minitorch 


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='mask, dropout, flash attention 2')
    parser.add_argument('--N', type=int)
    parser.add_argument('--flash2', action='store_true')
    parser.add_argument('--causal', action='store_true')
    args = parser.parse_args()

    kt.init(device="cuda:0", nhead=8)
    #batch_size = int(8192p/args.N)
    #batch_size = int(256/args.N)
    batch_size = 1
    for batch_size in [batch_size]:
        for nhead in [8]:
            for from_len in [args.N]: 
                for to_len  in [64]:
                    try:
                        kt.run('test_launch_flash_attn_bw')
                        print("", end="", flush=True)
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        pass
