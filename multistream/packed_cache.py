"""One-entry, process-local immutable CPU shard cache. Never caches GPU views."""
import dataclasses,hashlib,threading,weakref,os
import torch
from .log import log
LOCK=threading.RLock()
_entry=None

def version(t):
    try:return t._version
    except RuntimeError:return 'inference'

def storage_key(t):
    return (t.data_ptr(),version(t),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))

def tensor_key(t):
    if hasattr(t,'_qdata') and dataclasses.is_dataclass(getattr(t,'_params',None)):
        fields=[]
        for f in dataclasses.fields(t._params):
            value=getattr(t._params,f.name)
            if isinstance(value,torch.Tensor):
                # QuantizedTensor.detach clones small scale tensors. Compare their
                # exact CPU bytes, not wrapper/storage identity; never hash weights.
                if value.device.type=='cpu' and value.numel()*value.element_size()<=1024*1024:
                    raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    value=(tuple(value.shape),str(value.dtype),hashlib.sha256(raw).hexdigest())
                else:value=storage_key(value)
            fields.append((f.name,value))
        return ('quantized',t._layout_cls,storage_key(t._qdata),tuple(fields))
    return ('tensor',storage_key(t))

def signature(model,lora,generation):
    # CPU offload can rewrap quantized parameters while retaining the same data.
    # Hold source tensors in the entry so freed pointers cannot be recycled into
    # a false cache hit. Inference/.data mutations require cache_generation++.
    weights=tuple((name,tensor_key(t))
                  for name,t in list(model.blocks.named_parameters())+list(model.blocks.named_buffers()))
    adapter=() if not lora else (lora['strength'],lora['precision'],tuple(
        (name,tuple((id(t),version(t),tuple(t.shape),str(t.dtype)) for t in pair))
        for name,pair in sorted(lora['pairs'].items())))
    return (3,generation,weights,adapter,tuple(os.environ.get(k,d) for k,d in [('H3_TP_INTERNAL_OUTPUT','original'),('H3_TP_FC2_GEMM','vendor'),('H3_TP_SWIGLU_FUSION','joint')]))

def get(model,key):
    with LOCK:
        if _entry and _entry[0]() is model and _entry[1]==key:return _entry[2]
        if _entry:
            old=_entry[1]
            changes=[(a[0],a[1:3],b[1:3]) for a,b in zip(old[2],key[2]) if a!=b][:3]
            log.info('[H3 Load] cache miss diagnostic same_model=%s weight_changes=%s adapter_changed=%s generation_changed=%s',
                     _entry[0]() is model,changes,old[3]!=key[3],old[1]!=key[1])
        return None

def put(model,key,groups,max_gib=40):
    global _entry
    size=sum(rank.host.numel() if hasattr(rank,"host") else sum(g.host.numel() for g in rank) for rank in groups)
    if size>max_gib*2**30:
        log.warning('[H3 Load] CPU packed cache skipped: %.2fGiB > cap %.2fGiB',size/2**30,max_gib);return
    sources=tuple(t for _,t in list(model.blocks.named_parameters())+list(model.blocks.named_buffers()))
    with LOCK:_entry=(weakref.ref(model),key,groups,sources)
    log.info('[H3 Load] CPU packed cache stored %.3fGiB (one entry)',size/2**30)

def clear():
    global _entry
    with LOCK:_entry=None
