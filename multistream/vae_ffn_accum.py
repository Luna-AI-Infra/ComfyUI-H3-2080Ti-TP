"""VAE FFN-only FP16 accumulation. Never changes PyTorch global math flags."""
from .native_libs import library_path
import ctypes,os,threading
import torch
import torch.nn.functional as F
_LIB=None;_LOCK=threading.Lock();_LOCAL=threading.local();COUNTS={}
def mode():
 v=os.environ.get('H3_VAE_FFN_ACCUM','fp32')
 if v not in ('fp32','fp16'):raise ValueError('H3_VAE_FFN_ACCUM must be fp32/fp16')
 return v
def library():
 global _LIB
 with _LOCK:
  if _LIB is None:
   p=library_path('vae_half', 'H3_VAE_HALF_LIBRARY');l=ctypes.CDLL(p);V=ctypes.c_void_p;I=ctypes.c_int
   l.create.argtypes=[I,I,I,V];l.create.restype=V;l.run.argtypes=[V]*6;l.run.restype=I;l.destroy.argtypes=[V];_LIB=l
 return _LIB
class Plan:
 def __init__(self,m,n,k,b):
  self.lib=library();self.ptr=self.lib.create(m,n,k,b)
  if not self.ptr:raise RuntimeError('No cuBLASLt FP16-accumulation algorithm for VAE FFN')
 def __del__(self):
  if getattr(self,'ptr',None):self.lib.destroy(self.ptr)
def linear(x,w,b=None,output_fp32=False):
 policy=mode()
 if policy=='fp32' or x.dtype!=torch.float16 or w.dtype!=torch.float16:
  if output_fp32 and x.dtype==torch.float16:return torch.mm(x.reshape(-1,x.shape[-1]),w.T,out_dtype=torch.float32).view(*x.shape[:-1],w.shape[0])
  return F.linear(x,w,b)
 if not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(7,5):raise ValueError('FP16 VAE FFN backend validated only on SM75')
 if not w.is_contiguous() or (b is not None and (not b.is_contiguous() or b.dtype!=torch.float16)):raise ValueError('Unsupported VAE weight/bias layout')
 x=x.contiguous();m=x.numel()//x.shape[-1];n,k=w.shape
 if x.shape[-1]!=k or k%8 or n%8:raise ValueError('Unsupported VAE FFN shape')
 if not hasattr(_LOCAL,'cache'):_LOCAL.cache={};_LOCAL.workspace={}
 dev=x.device.index;key=(dev,m,n,k,b.data_ptr() if b is not None else 0)
 with torch.cuda.device(x.device):
  if key not in _LOCAL.cache:_LOCAL.cache[key]=Plan(m,n,k,None if b is None else b.data_ptr())
  if dev not in _LOCAL.workspace:_LOCAL.workspace[dev]=torch.empty(128<<20,device=x.device,dtype=torch.uint8)
  y=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.float16)
  code=library().run(_LOCAL.cache[key].ptr,x.data_ptr(),w.data_ptr(),y.data_ptr(),_LOCAL.workspace[dev].data_ptr(),torch.cuda.current_stream(x.device).cuda_stream)
  if code:raise RuntimeError(f'VAE FP16 accumulation failed: {code}')
 COUNTS[(dev,m,n,k)]=COUNTS.get((dev,m,n,k),0)+1
 return y.float() if output_fp32 else y
