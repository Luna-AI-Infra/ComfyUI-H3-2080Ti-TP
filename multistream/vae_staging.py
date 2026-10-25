"""Opt-in decoder-only CPU packing, last-use slot donation, and warm reuse.

Owns a private VAE, never mutates the input VAE or global model-management APIs.
Only the explicitly declared final Euler evaluation may donate a backbone slot.
"""
import concurrent.futures,copy,functools,hashlib,json,os,threading,time
from pathlib import Path
import torch
import comfy.sd,comfy.utils,comfy.model_management as mm
from .vae_tp import VAEBlockShard,VAETensorParallel
from .vae_precision import DEFAULT_MODE, TP_MODES, TOOLTIP, model_dtype, engine_kwargs, cast_shard_weights
from .profile_e35 import scope,copy_span
from .log import log

POOL=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='h3-vae-cpu-pack')
ENTRIES={};LOCK=threading.RLock()
FIELDS=('qw','qb','ow','ob','fw','fb','dw','db','n1','n2','s1','s2','qscale')

def key_for(vae,mode,generation=0):
    engine_kwargs(mode)
    factory=getattr(vae.patcher,'cached_patcher_init',None)
    if not factory or factory[0].__name__!='load_vae_patcher' or vae.patcher.patches:
        raise ValueError('VAE staging requires an unpatched checkpoint-backed H3 VAE')
    path=Path(factory[1][0]).resolve();st=path.stat()
    identity=(str(path),st.st_size,st.st_mtime_ns,st.st_ino,mode,generation,2)
    return hashlib.sha256(repr(identity).encode()).hexdigest(),path

class Pack:
    def __init__(self,blocks,boundary):
        self.templates=[copy.copy(b) for b in blocks];self.layout=[];offset=0
        for i,b in enumerate(blocks):
            for name in FIELDS:
                t=getattr(b,name);offset=(offset+255)//256*256;size=t.numel()*t.element_size()
                self.layout.append((i,name,offset,size,t.dtype,tuple(t.shape)));offset+=size
        self.boundary_layout=[]
        for name,t in boundary:
            offset=(offset+255)//256*256;size=t.numel()*t.element_size()
            self.boundary_layout.append((name,offset,size,t.dtype,tuple(t.shape)));offset+=size
        self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=True)
        for i,name,start,size,dtype,shape in self.layout:
            self.host[start:start+size].view(dtype).view(shape).copy_(getattr(blocks[i],name))
            setattr(self.templates[i],name,None)
        tensors=dict(boundary)
        for name,start,size,dtype,shape in self.boundary_layout:self.host[start:start+size].view(dtype).view(shape).copy_(tensors[name])
    def bind(self,buffer):
        blocks=[copy.copy(b) for b in self.templates]
        for i,name,start,size,dtype,shape in self.layout:setattr(blocks[i],name,buffer[start:start+size].view(dtype).view(shape))
        boundaries={name:buffer[start:start+size].view(dtype).view(shape) for name,start,size,dtype,shape in self.boundary_layout}
        return blocks,boundaries

class Entry:
    def __init__(self,path,mode,key):
        engine_kwargs(mode) # Reject unsupported policies before starting background work.
        self.path=path;self.mode=mode;self.key=key;self.lock=threading.RLock()
        self.gpu=[None,None];self.views=[None,None];self.ready=[None,None];self.streams=[None,None]
        self.boundaries={};self.local=None;self.packs=None;self.cpu_future=POOL.submit(self.build)
    def build(self):
        with scope('vae_staging/cpu_build'),torch.inference_mode(),torch.device('cpu'):
            sd=comfy.utils.load_torch_file(str(self.path),safe_load=True)
            # Constructor explicitly supports decoder-only H3 checkpoints.
            sd={k:v for k,v in sd.items() if k.startswith(('decoder.','post_quant_conv.')) or k in ('latents_mean','latents_std')}
            local=comfy.sd.VAE(sd=sd,dtype=model_dtype(self.mode));del sd
            model=local.first_stage_model;decoder=model.decoder
            boundary=[(n,p) for n,p in list(model.named_parameters())+list(model.named_buffers()) if not n.startswith('decoder.transformer_blocks.')]
            packs=[]
            for rank in (0,1):
                blocks=[VAEBlockShard(b,rank,torch.device('cpu'),model_dtype(self.mode)) for b in decoder.transformer_blocks]
                flags=engine_kwargs(self.mode)
                for b in blocks:cast_shard_weights(b,flags.get('balanced',False),flags.get('conservative',False))
                packs.append(Pack(blocks,boundary if rank==0 else []));del blocks
            self.local=local;self.packs=packs
            log.info('[H3 Load] VAE CPU pack ready GiB=%s',[round(p.host.numel()/2**30,4) for p in packs])
        return self
    def can_stage(self):
        if not self.cpu_future.done():return False
        self.cpu_future.result() # propagate failure; never silently use stale weights
        return True
    def stage_rank(self,rank,buffer=None,stream=None):
        self.cpu_future.result()
        with self.lock,torch.cuda.device(rank):
            if self.gpu[rank] is not None:return True
            pack=self.packs[rank]
            if buffer is None:
                if mm.get_free_memory(torch.device('cuda',rank))<pack.host.numel()+2*2**30:
                    raise RuntimeError('VAE staged shards need >=2GiB working reserve')
                buffer=torch.empty(pack.host.numel(),dtype=torch.uint8,device=rank)
            if buffer.numel()<pack.host.numel():return False
            stream=stream or torch.cuda.Stream(device=rank)
            with torch.cuda.stream(stream),copy_span('vae_staging/packed_h2d',torch.device('cuda',rank),stream,pack.host.numel()):
                buffer[:pack.host.numel()].copy_(pack.host,non_blocking=True)
                ready=torch.cuda.Event();ready.record(stream)
            views,boundaries=pack.bind(buffer)
            self.gpu[rank]=buffer;self.views[rank]=views;self.ready[rank]=ready;self.streams[rank]=stream
            if rank==0:self.boundaries=boundaries
            log.info('[H3 Load] VAE rank%d queued %.3fGiB on %s',rank,pack.host.numel()/2**30,'donated slot' if buffer.numel()>pack.host.numel() else 'new buffer')
            return True
    def release_gpu(self):
        with self.lock:
            for stream in self.streams:
                if stream is not None:stream.synchronize()
            self.gpu=[None,None];self.views=[None,None];self.ready=[None,None];self.streams=[None,None];self.boundaries={}
    def decode(self,latent,ffn_fusion=None):
        with scope('vae_staging/wait_cpu'):self.cpu_future.result()
        model=self.local.first_stage_model
        saved=[];engine=None
        try:
            with scope('vae_staging/load_remaining'):
                for rank in (0,1):self.stage_rank(rank)
                for rank in (0,1):
                    with torch.cuda.device(rank):torch.cuda.current_stream(rank).wait_event(self.ready[rank])
                for name,tensor in self.boundaries.items():
                    owner_name,_,field=name.rpartition('.');owner=model.get_submodule(owner_name) if owner_name else model
                    target=owner._parameters if field in owner._parameters else owner._buffers
                    saved.append((target,field,target[field]))
                    target[field]=torch.nn.Parameter(tensor,requires_grad=False) if target is owner._parameters else tensor
            engine=VAETensorParallel(model.decoder,ffn_fusion=ffn_fusion,**engine_kwargs(self.mode))
            engine.shards=self.views
            from .tp_comm import PeerExchange
            engine.ex=PeerExchange(engine.devices)
            with engine,scope('vae_staging/decode'),torch.inference_mode():
                # Same chunked H3 decoder, same temporal/spatial tiling and output
                # dtype as comfy.sd.VAE.decode. No all-model GPU reload needed.
                pixels=torch.empty(model.decode_output_shape(latent.shape),device=self.local.output_device,dtype=self.local.vae_output_dtype())
                for i in range(latent.shape[0]):
                    model.decode(latent[i:i+1].to(device=self.local.device,dtype=self.local.vae_dtype),output_buffer=pixels[i:i+1])
                    self.local.process_output(pixels[i:i+1])
                result=pixels.to(self.local.output_device).movedim(1,-1)
                if result.ndim==5:result=result.reshape(-1,*result.shape[-3:])
                for rank in (0,1):torch.cuda.synchronize(rank)
                return result
        finally:
            for target,field,tensor in saved:target[field]=tensor
            # Engine.__exit__ closes its worker pool; direct exceptions before
            # entering the context still need a pool cleanup.
            if engine is not None:engine.pool.shutdown(wait=True)
            self.release_gpu()

def obtain(vae,mode,generation=0):
    key,path=key_for(vae,mode,generation)
    with LOCK:
        entry=ENTRIES.get(key)
        if entry is None:
            # One private prepared decoder is retained; do not discard an active
            # entry. Single Comfy queue serializes normal sampler ownership.
            for old in ENTRIES.values():
                if not old.cpu_future.done() or any(x is not None for x in old.gpu):
                    raise RuntimeError('Previous VAE staging entry still in use')
            ENTRIES.clear();entry=Entry(path,mode,key);ENTRIES[key]=entry
            log.info('[H3 Load] VAE CPU cache MISS')
        else:log.info('[H3 Load] VAE CPU cache HIT')
        return entry

def donate(key,rank,bank,used_slot):
    entry=ENTRIES.get(key)
    if entry is None or not entry.can_stage():return False
    # All unused slots have reached their last use on the final group.
    # A small group may not fit the decoder: release obsolete slots first,
    # then allocate a decoder buffer, instead of silently losing overlap.
    candidates=[i for i,b in enumerate(bank.buffers) if i!=used_slot and b is not None]
    for other in candidates:
        buffer=bank.buffers[other]
        if buffer.numel()<entry.packs[rank].host.numel():continue
        with torch.cuda.device(bank.device),torch.cuda.stream(bank.stream):
            if bank.done[other] is not None:bank.stream.wait_event(bank.done[other])
            if not entry.stage_rank(rank,buffer,bank.stream):continue
        bank.views[other]=None;bank.loaded[other]=None;bank.buffers[other]=None
        return True
    # Avoid a device-wide synchronize. record_stream defers allocator reuse
    # until the copy stream's dependency on the last consumer has completed.
    with torch.cuda.device(bank.device),torch.cuda.stream(bank.stream):
        for other in candidates:
            if bank.done[other] is not None:bank.stream.wait_event(bank.done[other])
            bank.buffers[other].record_stream(bank.stream)
            bank.views[other]=None;bank.loaded[other]=None;bank.buffers[other]=None
        # Insufficient free space: safely defer decoder allocation to decode.
        if mm.get_free_memory(bank.device)<entry.packs[rank].host.numel()+2*2**30:
            return False
        return entry.stage_rank(rank,stream=bank.stream)


def install_failure_cleanup():
    import execution
    original=execution.get_output_data
    @functools.wraps(original)
    async def guarded(*args,**kwargs):
        try:return await original(*args,**kwargs)
        except BaseException:
            # Node failure/cancellation ends the lease; do not strand donated
            # memory when the downstream VAE node never gets to execute.
            for entry in list(ENTRIES.values()):entry.release_gpu()
            raise
    execution.get_output_data=guarded

class H3LoadPipeline:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'vae':('VAE',),'sigmas':('SIGMAS',),'sampler':('SAMPLER',),
        'enabled':('BOOLEAN',{'default':False}),'prefetch_last_euler_step':('BOOLEAN',{'default':False}),
        'precision_mode':(list(TP_MODES),{'default':DEFAULT_MODE,'tooltip':TOOLTIP}),
        'cache_generation':('INT',{'default':0,'min':0,'max':1000000})}}
    RETURN_TYPES=('MODEL',);FUNCTION='prepare';CATEGORY='H3/TP'
    def prepare(self,model,vae,sigmas,sampler,enabled=False,prefetch_last_euler_step=False,precision_mode=DEFAULT_MODE,cache_generation=0):
        if not enabled:return (model,)
        fn=getattr(sampler,'sampler_function',None)
        if prefetch_last_euler_step and (getattr(fn,'__name__','')!='sample_euler' or len(sigmas)<2 or float(sigmas[-1])!=0):
            raise ValueError('Slot donation is validated only for Euler with a zero final sigma')
        entry=obtain(vae,precision_mode,cache_generation)
        m=model.clone();opts=m.model_options.setdefault('transformer_options',{})
        opts['h3_vae_stage_key']=entry.key
        if prefetch_last_euler_step:opts['h3_vae_last_sigma']=float(sigmas[-2])
        return (m,)
