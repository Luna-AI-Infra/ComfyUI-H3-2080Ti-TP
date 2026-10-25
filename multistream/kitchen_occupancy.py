from .native_libs import library_path
import ctypes
from pathlib import Path
import torch
from h3_optimizations.native import int8_attention as K
lib=ctypes.CDLL(library_path('kitchen_shared_kv', 'H3_KITCHEN_SHARED_KV_LIBRARY'))
lib.launch.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p]
lib.launch.restype=ctypes.c_int
def run(carrier,route,mode,output_layout='hnd'):
    assert mode in (0,1,2) and carrier.cta_k==64 and route.q_tile==64 and route.kv_tile==32
    assert carrier.input_dtype==torch.float16 and carrier.original_head_dim==128
    output,dtype,geo,strides=K._attention_geometry(carrier,output_layout)
    b,nq,nk,h,hk,d=geo
    assert h==hk and route.encoding=='delta'
    lut=route.indices.contiguous();counts=route.counts.contiguous()
    vals=[b,h,nq,nk,h//hk,*strides,carrier.q_scale.stride(0),carrier.q_scale.stride(1),lut.shape[-1]]
    assert len(vals)==20
    info=(ctypes.c_int*20)(*vals)
    tensors=[carrier.q,carrier.k,carrier.v,output,carrier.q_scale,carrier.k_scale,carrier.v_scale,lut,counts]
    with torch.cuda.device(carrier.q.device):
        code=lib.launch(mode,*[t.data_ptr() for t in tensors],info,torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError(f'Experimental CUDA launch failed: {code}')
    return output
