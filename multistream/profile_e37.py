"""Opt-in host-only E2E timing; reversible without discarding warm caches.

No CUDA event spans, per-block scopes, Nsight or additional synchronizations
inside the measured requests. Existing sampler/decoder synchronization remains.
"""
import contextlib,os
from . import profile_e35 as P

def install(T):
    import execution,gc
    import comfy.model_management as mm
    import comfy.sd,comfy.utils
    from aiohttp import web
    from server import PromptServer
    from .vae_tp import VAETensorParallel
    from .tp_residency import TwoSlotBank
    from . import vae_staging as VS
    bindings=[(VS,'copy_span'),(execution,'get_output_data'),(gc,'collect'),(comfy.utils,'load_torch_file'),
              (comfy.sd.VAE,'__init__'),(comfy.sd.VAE,'decode'),(TwoSlotBank,'prefetch'),(T.BlockShard,'forward')]
    bindings += [(mm,n) for n in ('load_models_gpu','unload_all_models','free_memory','soft_empty_cache')]
    bindings += [(mm.LoadedModel,n) for n in ('model_load','model_unload')]
    bindings += [(T.ResidentTP,n) for n in ('prepare','run','release')]
    bindings += [(VAETensorParallel,n) for n in ('prepare','run','__exit__')]
    saved=[(obj,name,getattr(obj,name)) for obj,name in bindings]
    old_copy_span=P.copy_span
    os.environ['H3_TIMING_E35']=os.environ['H3_TIMING_E37']
    P.copy_span=lambda *args,**kwargs:contextlib.nullcontext()
    VS.copy_span=P.copy_span
    P.install(T)
    # Do not record individual blocks, change prefetch dependency ordering or
    # add CUDA events. Whole-step host scopes already wait for rank completion.
    for obj,name,original in saved:
        if (obj,name) in ((TwoSlotBank,'prefetch'),(T.BlockShard,'forward')):setattr(obj,name,original)
    state={'enabled':True,'measurement':'host scopes only; no GPU event/kernel profiling'}
    @PromptServer.instance.routes.post('/h3multistream/e37/finish_timing')
    async def finish(request):
        running,pending=PromptServer.instance.prompt_queue.get_current_queue()
        if running or pending:return web.json_response({'error':'queue must be idle'},status=409)
        if state['enabled']:
            P.flush()
            for obj,name,original in reversed(saved):setattr(obj,name,original)
            P.copy_span=old_copy_span;P._path=None;P._rows=[];P._copies=[];P._origins.clear()
            os.environ.pop('H3_TIMING_E35',None);os.environ.pop('H3_TIMING_E37',None)
            state['enabled']=False
        from . import qwen_stream,packed_cache,vae_staging
        return web.json_response(dict(state,qwen_cpu_cache_present=qwen_stream.CACHE is not None,
          backbone_cpu_cache_present=packed_cache._entry is not None,vae_cpu_cache_entries=len(vae_staging.ENTRIES)))
    return P.H3LoadBenchmarkNonce
