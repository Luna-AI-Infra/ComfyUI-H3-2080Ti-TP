# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in exact SM75 H3 FC2 INT8 GEMM; original quantization/epilogue preserved.

H3_TP_FC2_GEMM=sm75_tuned enables verified shape dispatch. Unset or 'vendor'
uses the original path. H3_TP_FC2_GEMM_LIBRARY selects the built C ABI library.
Unknown shapes/devices fall back. No split-K or LoRA-stream scheduling by default.
"""
from .native_libs import library_path
import contextlib,ctypes,logging,os,threading,copy
import torch
C=None

_POLICY={256:9,512:1,623:12,857:9,1024:9,1536:2,2048:2,3072:2,4096:2}
_LOCAL=threading.local();_LOCK=threading.Lock();_LIB=None;_ORIGINAL=None;_FAILED=False
_COUNTS={}
def counters():return dict(_COUNTS)

def _install():
 global _LIB,_ORIGINAL,_FAILED,C
 with _LOCK:
  if _LIB is not None:return True
  if _FAILED:return False
  path=library_path('fc2_int8', 'H3_TP_FC2_GEMM_LIBRARY')
  try:
   import comfy_kitchen.backends.cuda as C
   lib=ctypes.CDLL(path);lib.launch.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*5;lib.launch.restype=ctypes.c_int
  except (OSError,AttributeError,ImportError) as exc:
   logging.warning('H3 FC2 tuned library unavailable; using vendor: %s',exc);_FAILED=True;return False
  _ORIGINAL=C._C.cublas_gemm_int8;_LIB=lib;C._C.cublas_gemm_int8=_gemm
  return True

def _gemm(ac,wc,oc,workspace,stream):
 tile=getattr(_LOCAL,'tile',None)
 if tile is None:return _ORIGINAL(ac,wc,oc,workspace,stream)
 a,w,out=[torch.from_dlpack(v) for v in (ac,wc,oc)]
 valid=(a.is_cuda and w.device==a.device and out.device==a.device and a.dtype==w.dtype==torch.int8 and out.dtype==torch.int32 and a.ndim==2 and tuple(w.shape)==(2688,14336) and tuple(out.shape)==(a.shape[0],2688) and a.shape[1]==14336 and a.is_contiguous() and w.is_contiguous() and out.is_contiguous() and torch.cuda.get_device_capability(a.device)==(7,5))
 if not valid:return _ORIGINAL(C._wrap_for_dlpack(a),C._wrap_for_dlpack(w),C._wrap_for_dlpack(out),workspace,stream)
 code=_LIB.launch(tile,a.shape[0],2688,14336,1,a.data_ptr(),w.data_ptr(),out.data_ptr(),None,stream)
 if code:raise RuntimeError(f'H3 FC2 CUTLASS launch rejected: {code}; tile={tile}, M={a.shape[0]}')
 key=f'cuda:{a.device.index}/M{a.shape[0]}/tile{tile}';_COUNTS[key]=_COUNTS.get(key,0)+1

class _FC2:
 def __init__(self,original):self.original=original
 def __getattr__(self,name):return getattr(object.__getattribute__(self,'original'),name)
 def __setattr__(self,name,value):
  if name in ('weight','scale','bias','lora_down','lora_up') and 'original' in self.__dict__:setattr(self.original,name,value)
  else:object.__setattr__(self,name,value)
 def __copy__(self):return type(self)(copy.copy(self.original))
 def __call__(self,x,input_act=None):
  tile=_POLICY.get(x.shape[0]) if x.ndim==2 else None
  old=getattr(_LOCAL,'tile',None);_LOCAL.tile=tile
  try:return self.original(x,input_act=input_act)
  finally:_LOCAL.tile=old

def maybe_wrap(linear):
 mode=os.environ.get('H3_TP_FC2_GEMM','vendor')
 if mode!='sm75_tuned' or not linear.quantized or tuple(linear.weight.shape)!=(2688,14336):return linear
 if not _install():return linear
 return _FC2(linear)
