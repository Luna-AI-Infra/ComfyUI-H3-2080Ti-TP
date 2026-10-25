"""Opt-in host node wall timing; synchronize only at node boundaries."""
import functools,json,time,os
from pathlib import Path
def install():
 import execution,torch
 path=Path(os.environ['H3_NODE_TIMING_PATH']);path.parent.mkdir(parents=True,exist_ok=True)
 original=execution.get_output_data
 @functools.wraps(original)
 async def timed(prompt_id,unique_id,obj,*args,**kwargs):
  started=time.perf_counter();wall=time.time();ok=False
  try:
   result=await original(prompt_id,unique_id,obj,*args,**kwargs);ok=True;return result
  finally:
   if torch.cuda.is_initialized():
    for dev in range(torch.cuda.device_count()):torch.cuda.synchronize(dev)
   row=dict(prompt_id=prompt_id,node=str(unique_id),class_name=type(obj).__name__,started_epoch=wall,seconds=time.perf_counter()-started,success=ok)
   with path.open('a') as f:f.write(json.dumps(row)+'\n')
 execution.get_output_data=timed
