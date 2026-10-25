from .native_libs import library_path
from pathlib import Path
import ctypes,types,os
import torch
from h3_optimizations.native import int8_attention as K
from h3_optimizations.attention.sparse.router import SparseTileRouter,sort_selected_indices
lib=ctypes.CDLL(library_path('kitchen_offsets', 'H3_KITCHEN_OFFSETS_LIBRARY'))
lib.launch.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p];lib.launch.restype=ctypes.c_int
lib.pack_offsets.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*7+[ctypes.c_void_p];lib.pack_offsets.restype=ctypes.c_int
class OffsetRouter(SparseTileRouter):
 def _pack(self,source,g,m,selected=None):
  b,h=source.shape[:2];stride=(g.kv_tiles+3)//4*4
  lut=torch.empty((b,h,g.q_tiles,stride),device=source.device,dtype=torch.int32)
  counts=torch.empty((b,h,g.q_tiles),device=source.device,dtype=torch.int32)
  with torch.cuda.device(source.device):
   code=lib.pack_offsets(None if selected is None else selected.data_ptr(),lut.data_ptr(),counts.data_ptr(),b*h,g.q_tiles,g.kv_tiles,g.pure_video_q_start,g.pure_video_kv_start,m.retained_video_kv_tiles,stride,torch.cuda.current_stream().cuda_stream)
  if code:raise RuntimeError(code)
  return lut,counts,m
 def _dense_lut(self,source,g,m):return self._pack(source,g,m)
 def _build_lut_from_indices(self,source,g,m,indices):
  # Same top-k and sort as baseline; shift after sort preserves ordering exactly.
  selected=sort_selected_indices(indices.to(torch.int32))
  return self._pack(source,g,m,selected)
def make_route(lut,counts,offset=False):
 return types.SimpleNamespace(indices=lut,counts=counts,encoding='token_offset' if offset else 'delta',q_tile=64,kv_tile=32)
def run(carrier,route,mode,output_layout='hnd'):
 assert mode in range(5) and route.encoding==('token_offset' if mode in (1,3,4) else 'delta')
 assert carrier.cta_k==64 and carrier.input_dtype==torch.float16 and carrier.original_head_dim==128
 output,dtype,geo,strides=K._attention_geometry(carrier,output_layout);b,nq,nk,h,hk,d=geo
 assert h==hk and route.q_tile==64 and route.kv_tile==32
 lut=route.indices.contiguous();counts=route.counts.contiguous()
 vals=[b,h,nq,nk,h//hk,*strides,carrier.q_scale.stride(0),carrier.q_scale.stride(1),lut.shape[-1]]
 info=(ctypes.c_int*20)(*vals)
 tensors=[carrier.q,carrier.k,carrier.v,output,carrier.q_scale,carrier.k_scale,carrier.v_scale,lut,counts]
 with torch.cuda.device(carrier.q.device):code=lib.launch(mode,*[t.data_ptr() for t in tensors],info,torch.cuda.current_stream().cuda_stream)
 if code:raise RuntimeError(code)
 return output
