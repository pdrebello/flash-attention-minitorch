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

  ### OUR SOLUTION ####
  def flash_attention_minitorch():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    out_mt = q_mt.flash_attn(k_mt, v_mt, args.mask)

    start_time = time.time()
    #out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    #out_mt = minitorch.tensor(out.clone().tolist(), backend=backend, requires_grad=True)
    #l_mt = minitorch.tensor(l.clone().tolist(), backend=backend, requires_grad=True)
    #m_mt = minitorch.tensor(m.clone().tolist(), backend=backend, requires_grad=True)
      
    start_time = time.time()
    #dq,dk,dv = out_mt.lauch_flash_attn_bw(q_mt, k_mt, v_mt, out_mt, out_grad_mt, l_mt, m_mt)
    out_mt.backward(out_grad_mt)
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()

    return [
        dq,dk,dv
    ], end_time - start_time

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
    #mask = minitorch.tensor_from_numpy(np.zeros((from_len, from_len),dtype=datatype), backend=backend)
    res = (q_mt @ k_mt)/np.sqrt(to_len)
    if(args.mask):
        mask_mt = create_causal_mask(batch_size, nhead, from_len, to_len, q_mt.backend)
        res += mask_mt
    result = (drop * minitorch.nn.softmax(res , dim=3)) @ v_mt
    #result = (minitorch.softmax(((q_mt @ k_mt)/np.sqrt(to_len) + mask) , dim=3) * drop) @ v_mt

    start_time = time.time()
    result = result.sum()
    result.backward()
    end_time = time.time()
    dq = torch.tensor(q_mt.grad.to_numpy().astype(datatype)).cuda()
    dk = torch.tensor(k_mt.grad.to_numpy().transpose(0,1,3,2).astype(datatype)).cuda()
    dv = torch.tensor(v_mt.grad.to_numpy().astype(datatype)).cuda()

    
    return [dq, dk,dv], end_time - start_time
  return flash_attention_minitorch, attention_minitorch #flash_attention_torch #

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='mask, dropout, flash attention 2')
    parser.add_argument('--mask', action='store_true', help='test causal masking')
    parser.add_argument('--N', type=int)
    args = parser.parse_args()

    
    kt.init(device="cuda:0", nhead=8)
    for batch_size in [8]:
        for nhead in [8]:
            for from_len in [args.N]: 
                for to_len  in [64]:
                    #assert(to_len * 4 <= 1024)
                    try:
                        kt.run('test_launch_flash_attn_bw')
                        print("", end="", flush=True)
                    except:
                        pass
    
    """
    for batch_size in [128]:
        for nhead in [8]:
            for from_len in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for to_len  in [1, 2, 4, 8, 15]:
                    kt.run('test_launch_flash_attn_bw')
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for nhead in [1]:
            for from_len in [40]:
                for to_len  in [15]:
                    kt.run('test_launch_flash_attn_bw')
    """
    #kt.run(
    #  'test_launch_flash_attn_bw'
    #)
