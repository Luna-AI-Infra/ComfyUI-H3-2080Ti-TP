"""Opt-in shape and allocator measurements; no tensor-value copies or CUDA sync."""
import os,json,time,functools
from pathlib import Path
import torch
_path=None
def emit(kind,**data):
 if _path is not None:
  with (_path/'statistics.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),kind=kind,**data))+'\n')
def memory():
 rows=[]
 for i in range(torch.cuda.device_count()):
  s=torch.cuda.memory_stats(i)
  rows.append(dict(rank=i,allocated=torch.cuda.memory_allocated(i),reserved=torch.cuda.memory_reserved(i),max_allocated=torch.cuda.max_memory_allocated(i),max_reserved=torch.cuda.max_memory_reserved(i),retries=s.get('num_alloc_retries',0),ooms=s.get('num_ooms',0)))
 return rows
def install(T):
 import execution,comfy.sd
 from comfy.ldm.minimax.model import PackedLayout
 from . import profile_e35 as timing
 old_layout=PackedLayout.__init__
 @functools.wraps(old_layout)
 def layout(self,*a,**kw):
  old_layout(self,*a,**kw)
  emit('packed_layout',signature=self.signature,total_tokens=self.seq_len,segments=self.segments)
 PackedLayout.__init__=layout
 old_run=T.ResidentTP.run
 @functools.wraps(old_run)
 def run(self,dm,args,*a,**kw):
  emit('step_start',shape=list(args['img'].shape),memory=memory())
  result=old_run(self,dm,args,*a,**kw)
  rows=memory();emit('step_end',memory=rows)
  if any(r['retries'] or r['ooms'] for r in rows):raise RuntimeError('E53_MEMORY_STOP: CUDA allocation retry or OOM observed')
  return result
 T.ResidentTP.run=run
 for obj,name,label in [(comfy.sd.VAE,'encode','reference/vae_encode'),(comfy.sd.CLIP,'encode_from_tokens_scheduled','conditioning/qwen_encode')]:
  old=getattr(obj,name)
  def call(self,*a,_old=old,_label=label,**kw):
   with timing.scope(_label):return _old(self,*a,**kw)
  setattr(obj,name,call)
 old_node=execution.get_output_data
 @functools.wraps(old_node)
 async def node(prompt_id,unique_id,obj,*a,**kw):
  global _path
  path=Path(os.environ['H3_MEASURE_E53'])/str(prompt_id)
  if path!=_path:
   _path=path;path.mkdir(parents=True,exist_ok=True)
   for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i);torch.cuda.reset_accumulated_memory_stats(i)
  emit('node_start',node=str(unique_id),memory=memory())
  try:return await old_node(prompt_id,unique_id,obj,*a,**kw)
  finally:emit('node_end',node=str(unique_id),memory=memory())
 execution.get_output_data=node
