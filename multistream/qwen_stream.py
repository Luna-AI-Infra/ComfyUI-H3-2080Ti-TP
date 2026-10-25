"""Opt-in H3 Qwen immutable CPU masters, packed two-slot streaming.

Original GGUF weights and FP32 math are preserved. No model-manager registration
of temporary GPU views; original CPU Parameters are restored on every exit.
"""
import contextlib,copy,functools,threading,time,types,weakref
import torch
import comfy.model_management as mm
from .log import log
from . import profile_e35

LOCK=threading.RLock();CACHE=None;MISSING=object()

def native(t):
    return t.as_subclass(torch.Tensor) if type(t) is not torch.Tensor else t

def version(t):
    try:return t._version
    except RuntimeError:return 'inference'

def fingerprint(model,generation,nranks):
    return (generation,nranks,tuple((n,id(t),t.data_ptr(),version(t),tuple(t.shape),str(t.dtype),str(getattr(t,'tensor_type',None)))
            for n,t in list(model.named_parameters())+list(model.named_buffers())))

class Spec:
    def __init__(self,t,rows=None):
        if getattr(t,'patches',[]):raise ValueError('Qwen streaming does not support patched GGUF tensors')
        self.cls=type(t) if hasattr(t,'tensor_type') else None
        self.qtype=getattr(t,'tensor_type',None);self.shape=tuple(t.shape)
        raw=native(t).detach()
        if rows is not None:
            assert len(self.shape)==2
            raw=raw.reshape(self.shape[0],-1)[rows]
            self.shape=(raw.shape[0],self.shape[1])
        self.raw_shape=tuple(raw.shape);self.dtype=raw.dtype;self.cpu=raw.contiguous()
    def wrap(self,raw):
        if self.cls is None:return raw
        return self.cls(raw,tensor_type=self.qtype,tensor_shape=torch.Size(self.shape),patches=[])

class Pack:
    def __init__(self,items):
        self.layout=[];offset=0
        for name,spec in items:
            assert spec.cpu.device.type=='cpu'
            offset=(offset+255)//256*256;size=spec.cpu.numel()*spec.cpu.element_size()
            self.layout.append((name,spec,offset,size));offset+=size
        self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=True)
        for name,spec,start,size in self.layout:
            self.host[start:start+size].view(spec.dtype).reshape(spec.raw_shape).copy_(spec.cpu)
            spec.cpu=None
    def bind(self,buffer):
        return {name:spec.wrap(buffer[start:start+size].view(spec.dtype).reshape(spec.raw_shape)) for name,spec,start,size in self.layout}

def module_items(module):
    return [(n,Spec(t)) for n,t in list(module.named_parameters())+list(module.named_buffers())]

@contextlib.contextmanager
def bind_module(module,views):
    saved=[]
    try:
        for name,value in views.items():
            owner_name,_,field=name.rpartition('.');owner=module.get_submodule(owner_name) if owner_name else module
            mapping=owner._parameters if field in owner._parameters else owner._buffers
            saved.append((mapping,field,mapping[field]))
            mapping[field]=torch.nn.Parameter(value,requires_grad=False) if mapping is owner._parameters else value
        yield
    finally:
        for mapping,field,old in reversed(saved):mapping[field]=old

class Bank:
    def __init__(self,packs,device,session):
        self.packs=packs;self.device=device;self.session=session
        with torch.cuda.device(device):
            self.stream=torch.cuda.Stream(device=device)
            capacity=max(p.host.numel() for p in packs)
            if mm.get_free_memory(device)<2*capacity+2*2**30:raise RuntimeError('Qwen slots need 2GiB activation reserve')
            self.buffers=[torch.empty(capacity,dtype=torch.uint8,device=device) for _ in range(2)]
        self.loaded=[None,None];self.done=[None,None];self.ready=[None,None]
    def prefetch(self,index,slot):
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            if self.done[slot] is not None:self.stream.wait_event(self.done[slot])
            pack=self.packs[index]
            with self.session.copy_event('qwen/layer='+str(index),self.device,self.stream,pack.host.numel()):
                self.buffers[slot][:pack.host.numel()].copy_(pack.host,non_blocking=True)
            self.ready[slot]=torch.cuda.Event();self.ready[slot].record(self.stream)
        self.loaded[slot]=index
    def acquire(self,index):
        if index==0:
            self.prefetch(0,0);self.prefetch(1,1)
        slot=self.loaded.index(index)
        torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        if index+1<len(self.packs) and index+1 not in self.loaded:self.prefetch(index+1,1-slot)
        return slot,self.packs[index].bind(self.buffers[slot])
    def finish(self,slot):
        with torch.cuda.device(self.device):
            self.done[slot]=torch.cuda.Event();self.done[slot].record(torch.cuda.current_stream(self.device))
    def close(self):
        with torch.cuda.device(self.device):
            self.stream.synchronize();torch.cuda.synchronize(self.device)
        self.buffers=[]

class Entry:
    def __init__(self,transformer,nranks,key):
        self.owner=weakref.ref(transformer);self.key=key;self.nranks=nranks;self.visual_pack=None
        if nranks==1:self.packs=[[Pack(module_items(layer)) for layer in transformer.model.layers]]
        else:
            from .qwen_tp import make_packs
            self.packs=make_packs(transformer)
        self.bytes=sum(p.host.numel() for rank in self.packs for p in rank)

def obtain(transformer,nranks,generation):
    global CACHE
    key=fingerprint(transformer,generation,nranks)
    if CACHE is not None and CACHE.owner() is transformer and CACHE.key==key:return CACHE,True
    CACHE=None
    if any(t.device.type!='cpu' for t in transformer.parameters()):raise ValueError('Qwen CPU master must be entirely offloaded before staging')
    CACHE=Entry(transformer,nranks,key)
    log.info('[H3 Qwen] CPU packs %.3fGiB ranks=%d',CACHE.bytes/2**30,nranks)
    return CACHE,False

class Session:
    def __init__(self,transformer,entry,profile=False):
        self.transformer=transformer;self.entry=entry;self.profile=profile;self.devices=[torch.device('cuda',i) for i in range(entry.nranks)]
        self.banks=[];self.restores=[];self.copy_records=[];self.layer_records=[];self.origins={};self.stats={'nranks':entry.nranks,'packed_gib':entry.bytes/2**30,'origin_epoch':{}};self.visual_buffer=None
    @contextlib.contextmanager
    def copy_event(self,name,device,stream,nbytes):
        with profile_e35.copy_span(name,device,stream,nbytes),self.local_copy_event(name,device,stream,nbytes):yield
    @contextlib.contextmanager
    def local_copy_event(self,name,device,stream,nbytes):
        if not self.profile:yield;return
        with torch.cuda.device(device),torch.cuda.stream(stream):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record(stream)
            try:yield
            finally:b.record(stream);self.copy_records.append((name,device.index,nbytes,a,b))
    def override(self,module,name,value):
        self.restores.append((module,name,module.__dict__.get(name,MISSING)));setattr(module,name,value)
    def embedding(self,original,input,out_dtype=None):
        # Only selected embedding rows are needed. GGUF rows preserve complete
        # quantization blocks; there is no requantization or reduced precision.
        module=self.transformer.model.embed_tokens
        if input.numel()==0:return original(input,out_dtype=out_dtype)
        ids=input.detach().to('cpu',dtype=torch.long).reshape(-1)
        spec=Spec(module.weight,ids);raw=spec.cpu.pin_memory();device=input.device
        with self.copy_event('qwen/embedding_rows',device,torch.cuda.current_stream(device),raw.numel()*raw.element_size()):raw_gpu=raw.to(device,non_blocking=True)
        value=spec.wrap(raw_gpu)
        if hasattr(module,'get_weight'):value=module.get_weight(value,out_dtype)
        else:value=value.to(out_dtype)
        return native(value).reshape(*input.shape,-1).to(dtype=out_dtype)
    def visual(self,original,*args,**kwargs):
        if self.entry.visual_pack is None:self.entry.visual_pack=Pack(module_items(self.transformer.visual))
        pack=self.entry.visual_pack
        if self.visual_buffer is None:
            self.visual_buffer=torch.empty(pack.host.numel(),dtype=torch.uint8,device=self.devices[0])
            with self.copy_event('qwen/vision',self.devices[0],torch.cuda.current_stream(self.devices[0]),pack.host.numel()):self.visual_buffer.copy_(pack.host,non_blocking=True)
        with bind_module(self.transformer.visual,pack.bind(self.visual_buffer)):return original(*args,**kwargs)
    def single(self,index,original,*args,**kwargs):
        mm.throw_exception_if_processing_interrupted()
        bank=self.banks[0];slot,views=bank.acquire(index)
        if self.profile:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        with bind_module(self.transformer.model.layers[index],views):result=original(*args,**kwargs)
        if self.profile:b.record();self.layer_records.append((index,a,b))
        bank.finish(slot);return result
    def __enter__(self):
        if self.profile:
            for d in self.devices:
                with torch.cuda.device(d):
                    e=torch.cuda.Event(enable_timing=True);e.record();e.synchronize();self.origins[d.index]=e
                    self.stats['origin_epoch'][d.index]=time.time()
        try:
            self.banks=[Bank(p,d,self) for p,d in zip(self.entry.packs,self.devices)]
            embed=self.transformer.model.embed_tokens;self.override(embed,'forward',functools.partial(self.embedding,embed.forward))
            visual=self.transformer.visual;self.override(visual,'forward',functools.partial(self.visual,visual.forward))
            if self.entry.nranks==2:
                from .qwen_tp import Engine
                self.tp=Engine(self)
            for i,layer in enumerate(self.transformer.model.layers):
                fn=functools.partial(self.single,i,layer.forward) if self.entry.nranks==1 else functools.partial(self.tp.block,i)
                self.override(layer,'forward',fn)
            return self
        except BaseException:self.__exit__(*__import__('sys').exc_info());raise
    def __exit__(self,*exc):
        for module,name,original in reversed(self.restores):
            if original is MISSING:delattr(module,name)
            else:setattr(module,name,original)
        self.restores=[]
        if hasattr(self,'tp'):self.tp.close()
        for bank in self.banks:bank.close()
        self.banks=[];self.visual_buffer=None
        if self.profile:
            self.stats['copies']=[{'name':n,'rank':r,'bytes':size,'start_ms':self.origins[r].elapsed_time(a),'duration_ms':a.elapsed_time(b)} for n,r,size,a,b in self.copy_records]
            if self.entry.nranks==1:self.stats['layers']=[{'layer':i,'rank':0,'start_ms':self.origins[0].elapsed_time(a),'duration_ms':a.elapsed_time(b)} for i,a,b in self.layer_records]
        self.copy_records=[]

class H3QwenExecution:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'clip':('CLIP',),'mode':(['original','stream_single','stream_tp2'],{'default':'original'}),
        'cache_generation':('INT',{'default':0,'min':0,'max':1000000}),'profile_copy_events':('BOOLEAN',{'default':False})}}
    RETURN_TYPES=('CLIP',);FUNCTION='patch';CATEGORY='H3/TP'
    def patch(self,clip,mode='original',cache_generation=0,profile_copy_events=False):
        if mode=='original':return (clip,)
        if mode not in ('stream_single','stream_tp2'):raise ValueError(mode)
        if clip.patcher.patches or clip.patcher.forced_hooks is not None:raise ValueError('Qwen streaming requires unpatched conditioning weights')
        transformer=getattr(getattr(clip.cond_stage_model,'qwen3vl_32b',None),'transformer',None)
        if transformer is None or type(transformer).__name__!='MiniMaxQwen3VL':raise ValueError('Only H3 dedicated Qwen3-VL-32B is supported')
        if len(transformer.model.layers)!=50 or transformer.model.norm is not None:raise ValueError('Expected 50-layer unnormalized H3 encoder')
        own=clip.clone();nranks=2 if mode=='stream_tp2' else 1
        original_encode=own.encode_from_tokens.__func__
        def encode(self,*args,**kwargs):
            with LOCK,torch.inference_mode():
                t=time.perf_counter();mm.unload_all_models();unload=time.perf_counter()-t
                t=time.perf_counter()
                with profile_e35.scope('qwen/cpu_pack',nranks=nranks):entry,hit=obtain(transformer,nranks,cache_generation)
                prepare=time.perf_counter()-t
                session=Session(transformer,entry,profile_copy_events)
                t=time.perf_counter()
                with profile_e35.scope('qwen/encode_and_release',nranks=nranks),session:result=original_encode(self,*args,**kwargs)
                self.h3_qwen_last_stats=dict(session.stats,cpu_cache_hit=hit,prepare_seconds=prepare,previous_unload_seconds=unload,session_seconds=time.perf_counter()-t)
                log.info('[H3 Qwen] mode=%s cache=%s prepare=%.3fs encode/release=%.3fs',mode,hit,prepare,self.h3_qwen_last_stats['session_seconds'])
                if profile_e35._path is not None:
                    import json
                    profile_e35._path.mkdir(parents=True,exist_ok=True)
                    (profile_e35._path/'qwen-stats.json').write_text(json.dumps(self.h3_qwen_last_stats,indent=2))
                return result
        own.encode_from_tokens=types.MethodType(encode,own)
        own.load_model=types.MethodType(lambda self,tokens={}:self.patcher,own)
        return (own,)
