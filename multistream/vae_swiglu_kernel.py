from .native_libs import library_path
import ctypes,os,torch
lib=ctypes.CDLL(library_path('vae_swiglu', 'H3_VAE_SWIGLU_LIBRARY'))
lib.launch_vae_swiglu.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*4+[ctypes.c_void_p];lib.launch_vae_swiglu.restype=ctypes.c_int
def fused(x,vec=8,threads=128):
 assert vec in (2,4,8) and threads in (128,256)
 assert x.numel()<=2147483647 and x.data_ptr()%(2*vec)==0
 assert x.is_cuda and x.dtype==torch.float16 and x.is_contiguous() and x.shape[-1]%(2*vec)==0
 k=x.shape[-1]//2;m=x.numel()//(2*k);y=torch.empty((*x.shape[:-1],k),device=x.device,dtype=x.dtype)
 if m:
  with torch.cuda.device(x.device):code=lib.launch_vae_swiglu(x.data_ptr(),y.data_ptr(),m,k,vec,threads,torch.cuda.current_stream().cuda_stream)
  if code:raise RuntimeError(f'E49 CUDA launch failed {code}')
 return y
