"""Opt-in E64 end-to-end host ranges and Nsight CUDA trace; no extra sync."""
import contextlib,functools,json,os,threading,time
from pathlib import Path
import torch

def install(T):
    import execution,comfy.model_management as mm,comfy.sd,comfy.utils,gc
    from .vae_tp import VAETensorParallel
    from .tp_residency import TwoSlotBank
    P=Path(os.environ['H3_PROFILE_E64']);P.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();active=False;started=False;rows=[]
    @contextlib.contextmanager
    def scope(name):
        if not active:
            yield;return
        begin=time.perf_counter_ns();epoch=time.time_ns();tid=threading.get_native_id()
        torch.cuda.nvtx.range_push('E64/'+name)
        try:yield
        finally:
            torch.cuda.nvtx.range_pop()
            row=dict(name=name,start_ns=begin,end_ns=time.perf_counter_ns(),epoch_ns=epoch,tid=tid)
            with lock:
                rows.append(row)
    def wrap(obj,attr,name):
        original=getattr(obj,attr)
        @functools.wraps(original)
        def call(*a,**kw):
            with scope(name):return original(*a,**kw)
        setattr(obj,attr,call)
    for attr in ('load_models_gpu','unload_all_models','free_memory','soft_empty_cache'):
        wrap(mm,attr,'memory/'+attr)
    for attr in ('model_load','model_unload'):wrap(mm.LoadedModel,attr,'memory/'+attr)
    wrap(comfy.utils,'load_torch_file','weights/read_checkpoint')
    wrap(comfy.sd.VAE,'__init__','vae/construct')
    wrap(comfy.sd.VAE,'decode','vae/decode_inclusive')
    wrap(gc,'collect','memory/python_gc')
    for attr in ('prepare','run','release'):wrap(T.ResidentTP,attr,'backbone/'+attr)
    for attr in ('prepare','run','__exit__'):wrap(VAETensorParallel,attr,'vae_tp/'+attr)
    for attr in ('prefetch','acquire','finish','close'):wrap(TwoSlotBank,attr,'weight_bank/'+attr)
    from .tp_comm import PeerExchange
    for attr in ('start','finish','_copy'):
        wrap(PeerExchange,attr,'p2p/'+attr)
    for attr in ('encode',):wrap(comfy.sd.VAE,attr,'reference/vae_encode')
    wrap(comfy.sd.CLIP,'encode_from_tokens_scheduled','conditioning/qwen_encode')
    original_block=T.BlockShard.forward
    def block(self,rank,x,emb,segments,rope,options,exchange):
        old=self.stage
        self.stage=lambda name:scope(f'backbone/rank={rank}/block={options.get("block_index")}/{name}')
        try:
            with scope(f'backbone/rank={rank}/block={options.get("block_index")}'):
                return original_block(self,rank,x,emb,segments,rope,options,exchange)
        finally:self.stage=old
    T.BlockShard.forward=block
    VAETensorParallel.stage=lambda self,rank,name:scope(f'vae/rank={rank}/{name}')
    original=execution.get_output_data
    @functools.wraps(original)
    async def node(prompt_id,unique_id,obj,*a,**kw):
        nonlocal active,started
        if not started and (P/'ARM').exists():
            torch.cuda.cudart().cudaProfilerStart();active=True;started=True
            (P/'clock.json').write_text(json.dumps(dict(epoch_ns=time.time_ns(),perf_ns=time.perf_counter_ns())))
        try:
            with scope(f'node/{unique_id}/{type(obj).__name__}'):
                result=await original(prompt_id,unique_id,obj,*a,**kw)
        except BaseException as exc:
            if active:
                (P/'ranges.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (P/'capture-failed.json').write_text(json.dumps(dict(error=repr(exc),epoch_ns=time.time_ns())))
                try:torch.cuda.cudart().cudaProfilerStop()
                except BaseException:pass
                active=False
            raise
        if active and str(unique_id)=='19':
            for dev in (0,1):torch.cuda.synchronize(dev)
            torch.cuda.cudart().cudaProfilerStop();active=False
            (P/'ranges.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            from . import fc2_gemm
            (P/'fc2-dispatch.json').write_text(json.dumps(fc2_gemm.counters(),indent=2))
            (P/'capture-stopped.json').write_text(json.dumps(dict(epoch_ns=time.time_ns(),reason='save complete; both devices synchronized')))
        return result
    execution.get_output_data=node
