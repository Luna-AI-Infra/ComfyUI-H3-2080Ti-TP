"""E28 opt-in profiling only: live step2/block25 and complete video VAE."""
import contextlib,json,os,threading,time,collections
from pathlib import Path
import torch

def install(T):
    import nodes,comfy.ops
    from . import tp_lora as L
    from comfy.ldm.minimax import vae as V
    P=Path(os.environ['H3TP_PROFILE_E28']);P.mkdir(parents=True,exist_ok=True)
    barrier=threading.Barrier(2,timeout=180);counts=[0,0];local=threading.local()
    stats=collections.Counter();metadata={};original=T.BlockShard.forward
    @contextlib.contextmanager
    def scope(name):
        torch.cuda.nvtx.range_push(name)
        try:yield
        finally:torch.cuda.nvtx.range_pop()
    def forward(self,rank,x,emb,segments,rope,options,exchange):
        if options.get('block_index')!=25:return original(self,rank,x,emb,segments,rope,options,exchange)
        counts[rank]+=1
        if counts[rank]!=2:return original(self,rank,x,emb,segments,rope,options,exchange)
        if os.environ.get('H3TP_E28_INPUT_ONLY'):
            barrier.wait()
            if rank==0:
                layout=options['minimax_h3_layout']
                d=dict(x=x.detach().cpu(),emb=emb.detach().cpu(),rope=rope.detach().cpu(),segments=[(a,b,c.cpu() if isinstance(c,torch.Tensor) else c) for a,b,c in segments],packed=vars(layout))
                torch.save(d,P/'block25-input.pt')
            barrier.wait()
            return original(self,rank,x,emb,segments,rope,options,exchange)
        torch.cuda.synchronize(rank);barrier.wait()
        if rank==0:
            metadata['block']={'shape':list(x.shape),'dtype':str(x.dtype),'step_one_based':2,'block_zero_based':25,'lora_precision':self.qkv.lora_precision,'weight_residency':'three_groups','capture_boundary_synchronized':True}
            torch.cuda.cudart().cudaProfilerStart()
        barrier.wait();old=self.stage
        self.stage=lambda name:scope(f'E28/rank={rank}/phase={name}')
        local.block=True
        try:
            with scope(f'E28/rank={rank}/phase=block'):
                y=original(self,rank,x,emb,segments,rope,options,exchange)
                torch.cuda.synchronize(rank)
            barrier.wait()
            if rank==0:torch.cuda.cudart().cudaProfilerStop()
            barrier.wait()
            return y
        finally:
            self.stage=old;local.block=False
            if rank==0:(P/'capture-metadata.json').write_text(json.dumps(metadata,indent=2))
    T.BlockShard.forward=forward
    add=L.add_lora_
    def lora(*a,**kw):
        if getattr(local,'block',False):
            with scope('E28/LoRA_increment'):return add(*a,**kw)
        return add(*a,**kw)
    L.add_lora_=lora
    linear=comfy.ops.linear_input_act
    def lin(layer,x,*a,**kw):
        if not getattr(local,'vae',False):return linear(layer,x,*a,**kw)
        name=f'E28/VAE/linear/x={list(x.shape)}/w={list(layer.weight.shape)}/dtype={x.dtype}/act={a[0] if a else kw.get("input_act")}'
        stats[name]+=1
        with scope(name):return linear(layer,x,*a,**kw)
    comfy.ops.linear_input_act=lin
    attn=V.optimized_attention
    def attention(q,k,v,*a,**kw):
        if not getattr(local,'vae',False):return attn(q,k,v,*a,**kw)
        name=f'E28/VAE/attention_core/q={list(q.shape)}/dtype={q.dtype}/device={q.device}'
        stats[name]+=1
        with scope(name):return attn(q,k,v,*a,**kw)
    V.optimized_attention=attention
    decode=nodes.VAEDecode.decode
    def vae_decode(self,vae,samples):
        if not isinstance(vae.first_stage_model,V.MiniMaxH3VideoVAE):return decode(self,vae,samples)
        if os.environ.get('H3TP_E28_INPUT_ONLY'):
            torch.save(samples['samples'].detach().cpu(),P/'video-latent.pt')
            return decode(self,vae,samples)
        hooks=[];model=vae.first_stage_model
        def pre(name):
            def hook(module,args):
                shape=list(args[0].shape) if args and isinstance(args[0],torch.Tensor) else []
                tag=f'E28/VAE/module={name}/type={type(module).__name__}/shape={shape}'
                stats[tag]+=1;torch.cuda.nvtx.range_push(tag)
            return hook
        def post(module,args,output):
            torch.cuda.nvtx.range_pop()
            return None
        for name,module in model.named_modules():
            if name.startswith('decoder') and isinstance(module,(V.ViT3DDecoder,V.TransformerBlock,V.Attention,V.FeedForward,torch.nn.Linear,torch.nn.LayerNorm)):
                hooks.append(module.register_forward_pre_hook(pre(name)))
                hooks.append(module.register_forward_hook(post,always_call=True))
        metadata['vae']={'latent_shape':list(samples['samples'].shape),'device':str(vae.device),'dtype':str(vae.vae_dtype),'model_class':type(model).__name__,'decoder_layers':len(model.decoder.transformer_blocks),'profiling_includes_model_load':True,'tile_size':model.tile_size,'tokens_chunk_size':model.tokens_chunk_size,'attention_function':attn.__name__,'linear_types':{n:type(m).__name__ for n,m in model.decoder.named_modules() if isinstance(m,torch.nn.Linear)}}
        for r in range(2):torch.cuda.synchronize(r)
        local.vae=True;torch.cuda.cudart().cudaProfilerStart();begin=time.time()
        try:
            with scope('E28/VAE/whole_decode'):
                out=decode(self,vae,samples)
                for r in range(2):torch.cuda.synchronize(r)
            metadata['vae'].update(seconds=time.time()-begin,output_shape=list(out[0].shape))
            return out
        finally:
            torch.cuda.cudart().cudaProfilerStop();local.vae=False
            for hook in hooks:hook.remove()
            metadata['vae_ops']=dict(stats)
            (P/'capture-metadata.json').write_text(json.dumps(metadata,indent=2))
    nodes.VAEDecode.decode=vae_decode
