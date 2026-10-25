"""Opt-in VAE FP16 SwiGLU fusion, preserving both original FP16 rounding points."""
import os
import torch
import torch.nn.functional as F
MODES=('off','scalar','vector')
def resolve(mode=None):
 mode=os.environ.get('H3_VAE_SWIGLU_FUSION','off') if mode is None else mode
 if mode not in MODES:raise ValueError('H3_VAE_SWIGLU_FUSION must be off/scalar/vector')
 return mode
def activation(x,mode):
 # Mixed precision, unsupported layouts and other hardware retain original ops.
 if (mode=='off' or x.dtype!=torch.float16 or not x.is_cuda or not x.is_contiguous()
     or x.ndim<2 or x.shape[-1]%16 or x.data_ptr()%16
     or x.numel()>2147483647 or torch.cuda.get_device_capability(x.device)!=(7,5)):
  gate,value=x.chunk(2,-1);return F.silu(gate)*value
 if mode=='scalar':
  from .swiglu_fusion import fused
  y,_=fused(x.reshape(-1,x.shape[-1]),main=True,lora=False,threads=128)
  return y.view(*x.shape[:-1],x.shape[-1]//2)
 if mode=='vector':
  from .vae_swiglu_kernel import fused
  return fused(x,vec=8,threads=128)
 raise ValueError(mode)
