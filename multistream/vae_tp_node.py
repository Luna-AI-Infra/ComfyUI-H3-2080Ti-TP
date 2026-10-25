"""H3 video VAE selectable precision TP decoder; FP16 default, FP32 fallback."""
import gc,time,json
import torch
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .vae_tp import VAETensorParallel
from .vae_ffn_fusion import resolve as resolve_ffn_fusion, MODES as FFN_FUSION_MODES
from .vae_precision import DEFAULT_MODE, MODES, TOOLTIP, model_dtype, engine_kwargs

class H3TPVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'samples':('LATENT',),'vae':('VAE',),'precision_mode':(list(MODES),{'default':DEFAULT_MODE,'tooltip':TOOLTIP})},'optional':{'ffn_fusion':(list(FFN_FUSION_MODES),{'default':'off','tooltip':'FP16 VAE SwiGLU fusion. off: original; scalar: shared fusion; vector: vectorized fusion. Other precision policies keep the original activation.'}),'staged_load':('BOOLEAN',{'default':False}),'cache_generation':('INT',{'default':0,'min':0,'max':1000000})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='decode'
    CATEGORY='H3/TP'
    DESCRIPTION='H3 video VAE on CUDA 0/1 with selectable FP16, mixed, and FP32 policies. Original VAE weights are not modified.'
    def decode(self,vae,samples,precision_mode=DEFAULT_MODE,staged_load=False,cache_generation=0,ffn_fusion=None):
        ffn_fusion=resolve_ffn_fusion(ffn_fusion)
        dtype=model_dtype(precision_mode)
        if type(vae.first_stage_model).__name__!='MiniMaxH3VideoVAE':raise ValueError('H3TPVAEDecode requires the MiniMax H3 video VAE')
        if precision_mode.startswith('tp2_') and (torch.cuda.device_count()<2 or not torch.cuda.can_device_access_peer(0,1)):raise RuntimeError('TP VAE requires CUDA 0/1 with direct peer access; select single_fp32 to fall back')
        latent=samples['samples']
        if latent.is_nested:latent=latent.unbind()[0]
        stats={'mode':precision_mode,'ffn_fusion':ffn_fusion};start=time.perf_counter();local=None
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        old_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        old_accum=getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None)
        try:
            mm.unload_all_models()
            if staged_load:
                if not precision_mode.startswith('tp2_'):
                    raise ValueError('Staged loading requires a TP2 precision mode')
                from .vae_staging import obtain
                entry=obtain(vae,precision_mode,cache_generation)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
                if old_accum is not None:torch.backends.cuda.matmul.allow_fp16_accumulation=False
                stats['source']='prepared_decoder_shards'
                t=time.perf_counter();images=entry.decode(latent,ffn_fusion=ffn_fusion)
                stats['decode_including_wait_seconds']=time.perf_counter()-t
                return (images,)
            factory=getattr(vae.patcher,'cached_patcher_init',None)
            if factory and getattr(factory[0],'__name__','')=='load_vae_patcher' and not vae.patcher.patches:
                sd=comfy.utils.load_torch_file(factory[1][0],safe_load=True)
                stats['source']='loader_checkpoint'
            else:
                sd={k:v.detach().cpu() for k,v in vae.first_stage_model.state_dict().items()}
                stats['source']='model_state'
            local=comfy.sd.VAE(sd=sd,dtype=dtype);del sd
            mm.load_models_gpu([local.patcher]);torch.cuda.synchronize(0)
            stats['load_seconds']=time.perf_counter()-start
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
            if old_accum is not None:torch.backends.cuda.matmul.allow_fp16_accumulation=False
            with torch.inference_mode():
                if precision_mode.startswith('tp2_'):
                    with VAETensorParallel(local.first_stage_model.decoder,ffn_fusion=ffn_fusion,**engine_kwargs(precision_mode)) as engine:
                        t=time.perf_counter();engine.prepare()
                        for r in (0,1):torch.cuda.synchronize(r)
                        stats['shard_seconds']=time.perf_counter()-t;t=time.perf_counter();images=local.decode(latent)
                        for r in (0,1):torch.cuda.synchronize(r)
                        stats['decode_seconds']=time.perf_counter()-t
                else:
                    t=time.perf_counter();images=local.decode(latent);torch.cuda.synchronize(0);stats['decode_seconds']=time.perf_counter()-t
            if len(images.shape)==5:images=images.reshape(-1,*images.shape[-3:])
            return (images,)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=old_reduced
            if old_accum is not None:torch.backends.cuda.matmul.allow_fp16_accumulation=old_accum
            if local is not None:mm.unload_all_models()
            local=None;gc.collect();mm.soft_empty_cache()
            stats['node_seconds']=time.perf_counter()-start
            print('[H3TPVAEDecode] '+json.dumps(stats),flush=True)
