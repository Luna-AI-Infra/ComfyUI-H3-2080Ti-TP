# Experimental adapter, opt-in per instance. Baseline uses the production path.
import contextlib
import torch
from . import kitchen_offsets as E
class Attention:
 def __init__(self,base):
  self.base=base;self.mode=0;self.shared_kv=base.shared_kv
  self.router=E.OffsetRouter(q_tile=64,kv_tile=32);self.stream=None
 def __call__(self,q,k,v,options):
  if self.mode==0:return self.base(q,k,v,options)
  layout=self.base.convert_layout(options['minimax_h3_layout']);assert layout.seq_len==q.shape[-2]
  stage=options.get('h3_tp_sparse_stage',contextlib.nullcontext)
  with stage('sparse_cast'):q,k,v=[x.to(torch.float16).contiguous() for x in (q,k,v)]
  router=self.router if self.mode in (1,3) else self.base.router
  with stage('sparse_route'):lut,counts,meta=router.build_lut(q,k,layout,self.base.keep)
  with stage('sparse_quantize'):c=E.K.prequantize_int8_attention(q,k,v,cta_k=64)
  self.last_metadata=vars(meta)
  with stage('sparse_kernel'):out=E.run(c,E.make_route(lut,counts,self.mode in (1,3)),self.mode)
  return out.transpose(1,2)
