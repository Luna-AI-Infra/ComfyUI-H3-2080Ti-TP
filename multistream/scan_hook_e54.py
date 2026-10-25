"""E54 short real 50-block scan, same input per case, optional CUDA stream spans."""
import os,json,time,functools,threading
from pathlib import Path
import torch

def install(T):
 from .tp_residency import TwoSlotBank as B
 from . import profile_e53 as M
 P=Path(os.environ['H3_SCAN_E54']);P.mkdir(parents=True,exist_ok=True)
 original=T.ResidentTP.run
 old={n:getattr(B,n) for n in ['prefetch','acquire','finish']}
 events=[];enabled=[False];origins={}
 def event():
  e=torch.cuda.Event(enable_timing=True);e.record();return e
 def prefetch(self,g,s):
  if not enabled[0] or self.loaded[s]==g:return old['prefetch'](self,g,s)
  with torch.cuda.stream(self.stream):
   if self.done[s] is not None:self.stream.wait_event(self.done[s])
   a=event();r=old['prefetch'](self,g,s);b=event()
  events.append((self.device.index,'H2D',g,a,b,self.groups[g].host.numel()));return r
 def acquire(self,g,prefetch_next=True):
  if not enabled[0]:return old['acquire'](self,g,prefetch_next)
  a=event();r=old['acquire'](self,g,prefetch_next);b=event()
  events.append((self.device.index,'wait_ready',g,a,b,0));self._e54_start=b
  return r
 def finish(self,s):
  if enabled[0]:events.append((self.device.index,'compute_stream',self.current_group,self._e54_start,event(),0))
  return old['finish'](self,s)
 B.prefetch=prefetch;B.acquire=acquire;B.finish=finish
 @functools.wraps(original)
 def run(self,dm,args,user_patches,devices,*a,**kw):
  B.prefetch=prefetch;B.acquire=acquire;B.finish=finish
  n=args['img'].shape[0];label=f'n{n}';rows=[];reference=None
  cases=[('manual_groups',8,5),('auto_tokens',3,2)] if n<20000 else ([('manual_groups',5,2),('auto_tokens',3,2)] if n<35000 else [('auto_tokens',3,2),('manual_groups',8,3)])
  for mode,groups,slots in cases:
   self.release();self.weight_residency=mode;self.num_groups=groups;self.resident_slots=slots
   for d in devices:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(d);torch.cuda.reset_accumulated_memory_stats(d)
   row=dict(mode=mode,requested_groups=groups,requested_slots=slots,tokens=n,runs=[])
   rows.append(row);(P/(label+'.json')).write_text(json.dumps(rows,indent=2))
   for rep in range(3):
    events.clear();enabled[0]=rep==2
    for d in devices:
     with torch.cuda.device(d):origins[d.index]=event()
    epoch_start=time.time();begin=time.perf_counter()
    y=original(self,dm,args,user_patches,devices,*a,**kw)
    wall=time.perf_counter()-begin;enabled[0]=False
    mem=M.memory()
    row.update(actual_groups=len(self.group_ranges),actual_slots=self.active_slots,
       slot_bytes=[sum(b.numel() for b in bank.buffers if b is not None) for bank in self.banks],
       plan=getattr(self,'residency_plan',None) if mode=='auto_tokens' else None)
    spans=[dict(rank=r,kind=k,group=g,start_ms=origins[r].elapsed_time(x),end_ms=origins[r].elapsed_time(z),bytes=size) for r,k,g,x,z,size in events]
    item=dict(epoch_start=epoch_start,epoch_end=time.time(),repeat=rep,profiling=rep==2,wall_seconds=wall,memory=mem,spans=spans)
    row['runs'].append(item)
    # Outside timed scope; no trajectory mutation and no GPU full-size diff buffer.
    yc=y.detach().cpu();del y
    item['finite']=bool(torch.isfinite(yc).all())
    if reference is None:reference=yc
    else:
     diff=(yc-reference).double();item['max_abs']=float(diff.abs().max());item['relative_rmse']=float(diff.square().sum().sqrt()/reference.double().square().sum().sqrt());del diff
    (P/(label+'.json')).write_text(json.dumps(rows,indent=2))
    print('E54_CASE',label,mode,groups,slots,rep,wall,mem,flush=True)
    if not item['finite'] or any(x['retries'] or x['ooms'] for x in mem):
     raise RuntimeError('E54_MEMORY_STOP: nonfinite or allocator retry')
   self.release()
  raise RuntimeError('E54_SCAN_COMPLETE: controlled stop before sampler/VAE')
 T.ResidentTP.run=run
