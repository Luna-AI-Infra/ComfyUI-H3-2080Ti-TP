"""Experimental two-GPU tensor parallel H3 VAE decoder. Explicit context only."""
import contextlib,concurrent.futures,types
import torch
import torch.nn.functional as F
from .tp_comm import PeerExchange
from .vae_ffn_accum import linear as ffn_linear, mode as ffn_accum_mode
from .vae_precision import cast_shard_weights
from .vae_ffn_fusion import resolve as resolve_ffn_fusion, activation as ffn_activation
from comfy.ldm.minimax import vae as V
import comfy.ops

class VAEBlockShard:
    def __init__(self,b,rank,device,dtype):
        def copy(t):return t.detach().to(device=device,dtype=dtype,copy=True).contiguous()
        heads=b.attn.heads;hd=b.attn.dim_head;half=heads//2;inner=b.ff.w2.weight.shape[1];lo=rank*half;hi=lo+half
        rows=torch.arange(heads*3*hd).view(heads,3*hd)[lo:hi].flatten()
        self.qw=copy(b.attn.to_qkv.weight[rows]);self.qb=copy(b.attn.to_qkv.bias[rows])
        self.ow=copy(b.attn.to_out.weight[:,lo*hd:hi*hd]);self.ob=copy(b.attn.to_out.bias)
        fr=torch.cat([torch.arange(rank*inner//2,(rank+1)*inner//2),torch.arange(inner+rank*inner//2,inner+(rank+1)*inner//2)])
        self.fw=copy(b.ff.w1.weight[fr]);self.fb=copy(b.ff.w1.bias[fr]);self.dw=copy(b.ff.w2.weight[:,rank*inner//2:(rank+1)*inner//2]);self.db=copy(b.ff.w2.bias)
        self.n1=copy(b.norm1.weight);self.n2=copy(b.norm2.weight);self.s1=copy(b.scale1);self.s2=copy(b.scale2)
        self.eps=b.norm1.eps;self.qeps=b.attn.norm_q.eps;self.heads=half;self.hd=hd;self.qscale=torch.ones(hd,device=device,dtype=dtype)
    def partial(self,x,w):
        shape=x.shape[:-1]
        if x.dtype==torch.float16:return torch.mm(x.reshape(-1,x.shape[-1]),w.T,out_dtype=torch.float32).view(*shape,w.shape[0])
        return F.linear(x,w)

class VAETensorParallel:
    def __init__(self,decoder,profile=False,stable_fp16=False,balanced=False,conservative=False,attention_fp32=False,ffn_fusion=None):
        self.ffn_fusion=resolve_ffn_fusion(ffn_fusion)
        if ffn_accum_mode()=='fp16' and (balanced or stable_fp16):raise ValueError('FP16 FFN accumulation currently requires tp2_fp16')
        self.decoder=decoder;self.profile=profile;self.devices=[torch.device('cuda:0'),torch.device('cuda:1')]
        self.stable_fp16=stable_fp16;self.balanced=balanced;self.conservative=conservative;self.attention_fp32=attention_fp32
        if conservative and not balanced:raise ValueError('conservative requires balanced mode')
        if balanced and stable_fp16:raise ValueError('balanced and stable_fp16 are separate modes')
        self.original=decoder.forward;self.pool=concurrent.futures.ThreadPoolExecutor(2,thread_name_prefix='vae-tp')
        self.shards=None;self.ex=None;self.calls=0
    @contextlib.contextmanager
    def stage(self,rank,name):
        if self.profile:torch.cuda.nvtx.range_push(f'E29/rank={rank}/phase={name}')
        try:yield
        finally:
            if self.profile:torch.cuda.nvtx.range_pop()
    def prepare(self):
        dtype=self.decoder.x_embedder.weight.dtype
        if dtype not in (torch.float32,torch.float16):raise ValueError('SM75 VAE TP supports FP32/FP16 only')
        self.shards=[[],[]]
        # Read original CPU weights without holding a second complete GPU0 decoder.
        self.decoder.transformer_blocks.to('cpu')
        for r,dev in enumerate(self.devices):
            with torch.cuda.device(dev):self.shards[r]=[VAEBlockShard(b,r,dev,dtype) for b in self.decoder.transformer_blocks]
        if self.balanced:
            if dtype!=torch.float32:raise ValueError('balanced requires FP32 decoder boundary modules')
            for group in self.shards:
                for b in group:
                    cast_shard_weights(b,self.balanced,self.conservative)
        self.ex=PeerExchange(self.devices)
    def reduce(self,rank,value):
        rank,own,peer,done=self.ex.start(rank,value)
        torch.cuda.current_stream(self.devices[rank]).wait_event(done)
        return own+peer
    def run(self,h,rope):
        if self.shards is None:self.prepare()
        hs=[h,self.ex.copy_to(h,1)];ropes=[rope,self.ex.copy_to(rope,1)];ready=[]
        for dev in self.devices:
            e=torch.cuda.Event();e.record(torch.cuda.current_stream(dev));ready.append(e)
        def worker(rank):
            try:
                with torch.cuda.device(self.devices[rank]),torch.inference_mode(),self.stage(rank,'block'):
                    torch.cuda.current_stream(self.devices[rank]).wait_event(ready[rank]);x=hs[rank].float() if self.stable_fp16 else hs[rank]
                    for i,b in enumerate(self.shards[rank]):
                        with self.stage(rank,'qkv'):
                            n=F.rms_norm(x,(x.shape[-1],),b.n1.to(x.dtype),b.eps)
                            qkv=(b.partial(n.half(),b.qw)+b.qb) if self.balanced else F.linear(n.to(b.qw.dtype),b.qw,b.qb)
                            qkv=qkv.view(*x.shape[:2],b.heads,3*b.hd)
                            q,k,v=qkv.chunk(3,-1)
                        with self.stage(rank,'norm_rope'):
                            if self.stable_fp16:q,k=q.float(),k.float()
                            q,k=comfy.quant_ops.ck.rms_rope_split_half_(q,k,ropes[rank].to(q.dtype),b.qscale.to(q.dtype),epsilon=b.qeps,rot_dim=ropes[rank].shape[-3]*2)
                            if self.stable_fp16:q,k=q.to(v.dtype),k.to(v.dtype)
                        with self.stage(rank,'attention'):
                            if self.balanced and not self.attention_fp32:q,k,v=q.half(),k.half(),v.half()
                            y=V.optimized_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),b.heads,skip_reshape=True)
                            y=torch.nan_to_num(y)
                        with self.stage(rank,'out_projection'):part=b.partial(y.half() if self.balanced else y,b.ow)
                        with self.stage(rank,'attn_reduce'):out=self.reduce(rank,part)+b.ob.float()
                        with self.stage(rank,'residual_norm'):x=torch.addcmul(x.float(),out,b.s1.float()).to(x.dtype);n=F.rms_norm(x,(x.shape[-1],),b.n2.to(x.dtype),b.eps)
                        with self.stage(rank,'ffn_fc1_swiglu'):
                            fc1=(b.partial(n.half(),b.fw)+b.fb) if self.balanced and not self.conservative else ffn_linear(n.to(b.fw.dtype),b.fw,b.fb)
                            hidden=ffn_activation(fc1,self.ffn_fusion)
                        with self.stage(rank,'ffn_fc2'):part=ffn_linear(hidden.half() if self.balanced else hidden,b.dw,output_fp32=True)
                        with self.stage(rank,'ffn_reduce'):out=self.reduce(rank,part)+b.db.float()
                        with self.stage(rank,'residual'):x=torch.addcmul(x.float(),out,b.s2.float()).to(x.dtype)
                    torch.cuda.synchronize(rank)
                    return x
            except BaseException:
                self.ex.abort();raise
        futures=[self.pool.submit(worker,r) for r in (0,1)]
        results=[f.result() for f in futures];self.calls+=1
        return results[0]
    def forward(self,x):
        d=self.decoder;B,C,T,H,W=x.shape
        h=d.x_embedder(x.flatten(2).transpose(1,2));n=h.shape[1]
        h=torch.cat([h,comfy.ops.cast_to_input(d.register_tokens,h).expand(B,-1,-1),torch.zeros_like(h[:,:1])],1)
        ids=V.create_token_ids((T,H,W),x.device,x.dtype).expand(B,-1,-1)
        ids=torch.cat([ids,torch.zeros((B,1+d.num_register_tokens,3),device=x.device,dtype=ids.dtype)],1)
        h=self.run(h,d.pos_embed(ids))
        if self.stable_fp16:
            norm=d.norm_out
            h=F.layer_norm(h,norm.normalized_shape,norm.weight.to(h.dtype) if norm.weight is not None else None,norm.bias.to(h.dtype) if norm.bias is not None else None,norm.eps)
        else:
            h=d.norm_out(h)
        out=d.proj_out(h.to(d.x_embedder.weight.dtype))[:,:n]
        out=out.view(B,T,H,W,d.out_channels,d.patch_size_t,d.patch_size,d.patch_size)
        return out.permute(0,4,1,5,2,6,3,7).contiguous().reshape(B,d.out_channels,T*d.patch_size_t,H*d.patch_size,W*d.patch_size)
    def __enter__(self):
        self.decoder.forward=self.forward;return self
    def __exit__(self,*exc):
        self.decoder.forward=self.original
        self.pool.shutdown(wait=True)
        if self.ex:self.ex.close()
        self.shards=None;self.ex=None
