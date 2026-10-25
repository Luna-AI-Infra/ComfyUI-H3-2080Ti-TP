"""Opt-in host ranges and CUDA memcpy events; no CUPTI or per-kernel tracing."""
import contextlib,functools,json,os,threading,time
from pathlib import Path
import torch
_lock=threading.Lock();_rows=[];_copies=[];_path=None;_origins={}

@contextlib.contextmanager
def scope(name,**extra):
    if _path is None: yield;return
    a=time.perf_counter_ns();epoch=time.time_ns()
    try: yield
    finally:
        with _lock:_rows.append(dict(name=name,start_ns=a,end_ns=time.perf_counter_ns(),epoch_ns=epoch,tid=threading.get_native_id(),**extra))

@contextlib.contextmanager
def copy_span(name,device,stream,nbytes):
    if _path is None: yield;return
    dev=torch.device(device);index=dev.index
    with torch.cuda.device(dev),torch.cuda.stream(stream):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        try:yield
        finally:
            b.record()
            with _lock:_copies.append((name,index,nbytes,a,b))

def flush():
    if _path is None:return
    _path.mkdir(parents=True,exist_ok=True)
    with _lock:
        (_path/'ranges.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in _rows))
        out=[]
        for name,dev,size,a,b in _copies:
            if b.query():
                origin,epoch=_origins[dev]
                out.append(dict(name=name,rank=dev,bytes=size,start_epoch=epoch+origin.elapsed_time(a)/1000,
                                duration_ms=a.elapsed_time(b)))
        (_path/'copies.json').write_text(json.dumps(out,indent=2))

def install(T):
    import execution,comfy.model_management as mm,comfy.sd,comfy.utils,gc
    from .vae_tp import VAETensorParallel
    from .tp_residency import TwoSlotBank
    for r in (0,1):
        with torch.cuda.device(r):
            e=torch.cuda.Event(enable_timing=True);e.record();e.synchronize();_origins[r]=(e,time.time())
    def wrap(obj,attr,name):
        original=getattr(obj,attr)
        @functools.wraps(original)
        def call(*a,**kw):
            with scope(name):return original(*a,**kw)
        setattr(obj,attr,call)
    for attr in ('load_models_gpu','unload_all_models','free_memory','soft_empty_cache'):wrap(mm,attr,'memory/'+attr)
    for attr in ('model_load','model_unload'):wrap(mm.LoadedModel,attr,'memory/'+attr)
    wrap(comfy.utils,'load_torch_file','weights/read_checkpoint');wrap(comfy.sd.VAE,'__init__','vae/construct')
    wrap(comfy.sd.VAE,'decode','vae/decode_inclusive');wrap(gc,'collect','memory/python_gc')
    for attr in ('prepare','run','release'):wrap(T.ResidentTP,attr,'backbone/'+attr)
    for attr in ('prepare','run','__exit__'):wrap(VAETensorParallel,attr,'vae_tp/'+attr)
    prefetch=TwoSlotBank.prefetch
    def cp(self,group,slot):
        if self.loaded[slot]==group:return prefetch(self,group,slot)
        # Record after dependency event, so duration excludes waiting for slot use.
        with torch.cuda.stream(self.stream):
            if self.done[slot] is not None:self.stream.wait_event(self.done[slot])
            with copy_span(f'backbone/group={group}',self.device,self.stream,self.groups[group].host.numel()):
                return prefetch(self,group,slot)
    TwoSlotBank.prefetch=cp
    original_block=T.BlockShard.forward
    def block(self,rank,x,emb,segments,rope,options,exchange):
        with scope(f'backbone/rank={rank}/block={options.get("block_index")}'):
            return original_block(self,rank,x,emb,segments,rope,options,exchange)
    T.BlockShard.forward=block
    original=execution.get_output_data
    @functools.wraps(original)
    async def node(prompt_id,unique_id,obj,*a,**kw):
        global _path,_rows,_copies
        path=Path(os.environ['H3_TIMING_E35'])/str(prompt_id)
        if _path!=path:
            flush();_path=path;_rows=[];_copies=[]
        with scope(f'node/{unique_id}/{type(obj).__name__}'):
            try:result=await original(prompt_id,unique_id,obj,*a,**kw)
            except BaseException:flush();raise
        if str(unique_id)=='19':flush()
        return result
    execution.get_output_data=node

class H3LoadBenchmarkNonce:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'clip':('CLIP',),'nonce':('INT',{'default':0})}}
    RETURN_TYPES=('CLIP',);FUNCTION='pass_through';CATEGORY='H3/experiments'
    def pass_through(self,clip,nonce):return (clip,)
