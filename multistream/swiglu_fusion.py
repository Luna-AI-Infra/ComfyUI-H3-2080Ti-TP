"""Explicit E45 fusion. Joint fusion is the production default; explicit off restores the original path."""
from .native_libs import library_path
import ctypes
import copy
import os
import threading
import torch
import torch.nn.functional as F

_lib = None
_lock = threading.Lock()
MODES = ('off', 'main', 'lora', 'joint')

def library():
    global _lib
    with _lock:
        if _lib is None:
            path = library_path('swiglu', 'H3_TP_SWIGLU_LIBRARY')
            lib = ctypes.CDLL(path)
            lib.launch_swiglu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*5 + [ctypes.c_void_p]
            lib.launch_swiglu.restype = ctypes.c_int
            _lib = lib
    return _lib

def fused(x, main=True, lora=True, threads=256):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.float32) or not x.is_contiguous() or x.ndim != 2 or x.shape[1] % 2:
        raise ValueError('E45 requires contiguous CUDA FP16/FP32 [M,2K]')
    if not main and not lora: raise ValueError('No output requested')
    m, k = x.shape[0], x.shape[1]//2
    y = torch.empty((m,k),device=x.device,dtype=x.dtype) if main else None
    # FP32 main and LoRA have identical arithmetic, so share one output.
    alias = main and lora and x.dtype == torch.float32
    z = y if alias else torch.empty((m,k),device=x.device,dtype=torch.float32) if lora else None
    if m:
        with torch.cuda.device(x.device):
            status = library().launch_swiglu(x.data_ptr(), y.data_ptr() if main else None,
                z.data_ptr() if lora and not alias else None, m,k,0 if x.dtype==torch.float16 else 1,
                int(main) + 2*int(lora and not alias),threads,torch.cuda.current_stream(x.device).cuda_stream)
        if status: raise RuntimeError(f'E45 CUDA launch failed: {status}')
    return y,z

class FC2:
    """Wrap E42 HalfOutputLinear or the original LinearShard, preserving LoRA policy."""
    def __init__(self, original, Q, L, mode='off'):
        if mode not in MODES: raise ValueError(mode)
        self.original,self.Q,self.L,self.mode = original,Q,L,mode
        self.half_policy = hasattr(original,'base')
        self.base = original.base if self.half_policy else original
        self.threads = 256

    def __call__(self,x,input_act=None):
        b=self.base
        if (self.mode=='off' or input_act!='swiglu' or not b.quantized or
            tuple(b.weight.shape)!=(2688,14336) or x.shape[-1]!=28672 or
            x.dtype not in (torch.float16,torch.float32) or x.ndim!=2 or
            not x.is_contiguous() or not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(7,5)):
            return self.original(x,input_act)
        has_lora=b.lora_down is not None and b.lora_strength!=0
        # Original half LinearShard rounds before LoRA; only the E42 adapter
        # explicitly computes the LoRA activation from FP32 converted inputs.
        if not self.half_policy and x.dtype!=torch.float32:
            return self.original(x,input_act)
        if has_lora and not self.half_policy and b.lora_precision!='fp32':
            return self.original(x,input_act)
        do_main=self.mode in ('main','joint')
        do_lora=has_lora and self.mode in ('lora','joint')
        if not do_main and not do_lora:return self.original(x,input_act)
        act,source=fused(x,do_main,do_lora,self.threads)
        dtype=torch.float16 if self.half_policy else x.dtype
        out=self.Q.ck.int8_linear(act if do_main else x,b.weight,b.scale,
            None if b.bias is None else b.bias.to(dtype),dtype,
            convrot=b.convrot,convrot_groupsize=b.groupsize,input_act=None if do_main else 'swiglu')
        if has_lora:
            if not do_lora:
                source=x.float() if self.half_policy else x
                gate,value=source.chunk(2,-1);source=F.silu(gate)*value
            out=self.L.add_lora_(out.float() if self.half_policy else out,source,b.lora_down,b.lora_up,b.lora_strength,'fp32')
            if self.half_policy:out=out.half()
        return out

    def __getattr__(self,name):return getattr(object.__getattribute__(self,'original'),name)

    def __setattr__(self,name,value):
        if name in ('weight','scale','bias','lora_down','lora_up') and 'base' in self.__dict__:
            setattr(self.base,name,value)
        else:object.__setattr__(self,name,value)

    def __copy__(self):
        original=copy.copy(self.original)
        if self.half_policy:original.base=copy.copy(self.base)
        result=type(self)(original,self.Q,self.L,self.mode)
        result.threads=self.threads
        return result

def maybe_wrap(original,Q,L):
    mode=os.environ.get('H3_TP_SWIGLU_FUSION','joint')
    if mode not in MODES:raise ValueError('H3_TP_SWIGLU_FUSION must be off/main/lora/joint')
    return original if mode=='off' else FC2(original,Q,L,mode)
