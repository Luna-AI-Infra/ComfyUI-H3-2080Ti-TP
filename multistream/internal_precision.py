"""Opt-in INT8 linear FP16 outputs; residual/AdaLN boundaries stay FP32."""
import copy,os
import torch
import torch.nn.functional as F
FIELDS=('weight','scale','bias','lora_down','lora_up')
class HalfOutputLinear:
 def __init__(self,base,Q,L):self.base,self.Q,self.L=base,Q,L
 def __getattr__(self,name):return getattr(object.__getattribute__(self,'base'),name)
 def __setattr__(self,name,value):
  if name in FIELDS and 'base' in self.__dict__:setattr(self.base,name,value)
  else:object.__setattr__(self,name,value)
 def __copy__(self):return type(self)(copy.copy(self.base),self.Q,self.L)
 def __call__(self,x,input_act=None):
  b=self.base
  out=self.Q.ck.int8_linear(x.contiguous(),b.weight,b.scale,None if b.bias is None else b.bias.half(),torch.float16,convrot=b.convrot,convrot_groupsize=b.groupsize,input_act=input_act)
  if b.lora_down is not None and b.lora_strength:
   source=x.float()
   if input_act=='swiglu':
    g,v=source.chunk(2,-1);source=F.silu(g)*v
   out=self.L.add_lora_(out.float(),source,b.lora_down,b.lora_up,b.lora_strength,'fp32').half()
  return out
def mode():
 value=os.environ.get('H3_TP_INTERNAL_OUTPUT','original')
 if value not in ('original','ffn_fp16','linears_fp16'):raise ValueError('Invalid H3_TP_INTERNAL_OUTPUT')
 return value
def wrap(shard,Q,L):
 policy=mode()
 for name in (('fc1','fc2') if policy=='ffn_fp16' else ('qkv','out','fc1','fc2') if policy=='linears_fp16' else ()):
  b=getattr(shard,name)
  if not b.quantized:raise ValueError('FP16 internal output policy requires INT8 shards')
  setattr(shard,name,HalfOutputLinear(b,Q,L))
