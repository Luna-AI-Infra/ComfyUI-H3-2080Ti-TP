"""H3 Qwen TP2: GGUF row shards, full-K GEMMs, direct event P2P gathers.

Language conditioning only; visual/deepstack orchestration stays in upstream.
No precision conversion beyond the original layer's get_weight path.
"""
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.ldm.common_dit import rms_norm
from comfy.text_encoders.llama import apply_rope
from .qwen_stream import Pack,Spec,native
from .tp_comm import PeerExchange

PROJECTIONS={'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
             'mlp.gate_proj','mlp.up_proj','mlp.down_proj'}

def make_packs(transformer):
    result=[[],[]]
    for layer in transformer.model.layers:
        att=layer.self_attn
        if att.merged_qkv or layer.mlp.merged_mlp or att.num_heads%2 or att.num_kv_heads%2:
            raise ValueError('Qwen TP2 requires separate projections and even Q/KV heads')
        for rank in (0,1):
            items=[]
            for name,t in list(layer.named_parameters())+list(layer.named_buffers()):
                projection,_,field=name.rpartition('.')
                if projection in PROJECTIONS:
                    if field!='weight' or len(t.shape)!=2:raise ValueError('Only bias-free Qwen projections supported')
                    if t.shape[0]%2:raise ValueError('Odd output rows cannot be TP2 sharded')
                    half=t.shape[0]//2;spec=Spec(t,slice(rank*half,(rank+1)*half))
                else:spec=Spec(t)
                items.append((name,spec))
            result[rank].append(Pack(items))
    return result

class Engine:
    def __init__(self,session):
        self.session=session;self.devices=session.devices;self.exchange=PeerExchange(self.devices)
        self.streams=[torch.cuda.Stream(device=d) for d in self.devices]
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='h3-qwen-tp')
        self.static={};self.events=[];self.communication_bytes=[0,0]
    def move(self,value,rank):
        if isinstance(value,torch.Tensor):return self.exchange.copy_to(value,rank)
        if isinstance(value,tuple):return tuple(self.move(v,rank) for v in value)
        if isinstance(value,list):return [self.move(v,rank) for v in value]
        return value
    def static_move(self,value,rank):
        key=(id(value),rank)
        if key not in self.static:self.static[key]=(value,self.move(value,rank))
        return self.static[key][1]
    def gather(self,rank,value):
        self.communication_bytes[rank]+=value.numel()*value.element_size()
        return self.exchange.gather(rank,value)
    def run(self,rank,index,x,mask,freqs,attention,ready):
        device=self.devices[rank];stream=self.streams[rank]
        try:
            with torch.inference_mode(),torch.cuda.device(device),torch.cuda.stream(stream):
                stream.wait_event(ready)
                bank=self.session.banks[rank];slot,w=bank.acquire(index)
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                if self.session.profile:start.record(stream)
                layer=self.session.transformer.model.layers[index];att=layer.self_attn
                def norm(name,value):
                    module=layer.get_submodule(name);weight=native(w[name+'.weight'])
                    return rms_norm(value,weight+1.0 if module.add else weight,module.eps)
                def linear(name,value):
                    module=layer.get_submodule(name);weight=w[name+'.weight']
                    weight=module.get_weight(weight,value.dtype) if hasattr(module,'get_weight') else weight.to(value.dtype)
                    return F.linear(value,native(weight))
                batch,seq,_=x.shape;residual=x;x=norm('input_layernorm',x)
                heads=att.num_heads//2;kv=att.num_kv_heads//2;dim=att.head_dim
                q=linear('self_attn.q_proj',x).view(batch,seq,heads,dim).transpose(1,2)
                k=linear('self_attn.k_proj',x).view(batch,seq,kv,dim).transpose(1,2)
                v=linear('self_attn.v_proj',x).view(batch,seq,kv,dim).transpose(1,2)
                if att.q_norm is not None:q=norm('self_attn.q_norm',q)
                if att.k_norm is not None:k=norm('self_attn.k_norm',k)
                q,k=apply_rope(q,k,freqs_cis=freqs)
                context=attention(q,k,v,heads,mask=mask,skip_reshape=True,**({'enable_gqa':True} if heads!=kv else {}))
                context=self.gather(rank,context)
                x=residual+self.gather(rank,linear('self_attn.o_proj',context))
                residual=x;x=norm('post_attention_layernorm',x)
                gate=linear('mlp.gate_proj',x);up=linear('mlp.up_proj',x)
                activated=self.gather(rank,layer.mlp.activation(gate)*up)
                x=residual+self.gather(rank,linear('mlp.down_proj',activated))
                bank.finish(slot)
                done=torch.cuda.Event();done.record(stream)
                if self.session.profile:
                    end.record(stream);self.events.append((index,rank,start,end))
                return x,done
        except BaseException:self.exchange.abort();raise
    def block(self,index,x,attention_mask=None,freqs_cis=None,optimized_attention=None,past_key_value=None):
        mm.throw_exception_if_processing_interrupted()
        if past_key_value is not None:raise ValueError('Qwen TP2 supports conditioning, not autoregressive KV caches')
        inputs=[]
        for rank,device in enumerate(self.devices):
            with torch.cuda.device(device):
                value=self.move(x,rank);mask=self.static_move(attention_mask,rank);freqs=self.static_move(freqs_cis,rank)
                ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device))
                inputs.append((rank,index,value,mask,freqs,optimized_attention,ready))
        futures=[self.pool.submit(self.run,*args) for args in inputs]
        results=[future.result() for future in futures]
        for rank,(value,done) in enumerate(results):
            torch.cuda.current_stream(self.devices[rank]).wait_event(done)
            value.record_stream(torch.cuda.current_stream(self.devices[rank]))
        return results[0][0],None
    def close(self):
        self.pool.shutdown(wait=True)
        self.exchange.close()
        for stream in self.streams:stream.synchronize()
        self.session.stats['p2p_gather_bytes_per_rank']=self.communication_bytes
        if self.session.profile:
            self.session.stats['layers']=[{'layer':i,'rank':r,'start_ms':self.session.origins[r].elapsed_time(a),'duration_ms':a.elapsed_time(b)} for i,r,a,b in self.events]
        self.static.clear()
