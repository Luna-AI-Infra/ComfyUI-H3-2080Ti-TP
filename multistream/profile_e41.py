"""Opt-in E2E host timing with separately switchable CUDA stream spans.

Stream spans include GPU waits and inter-kernel gaps; they are not kernel-active
time. No precision or scheduling decisions change, and ordinary runs add no
per-group CUDA events. Disable event capture before E37 finish_timing.
"""
import os
from . import profile_e35 as P
from . import profile_e37 as H

def install(T):
    from aiohttp import web
    from server import PromptServer
    from . import vae_staging as VS
    from .tp_residency import TwoSlotBank as Bank
    import torch
    real_copy_span=P.copy_span
    os.environ['H3_TIMING_E37']=os.environ['H3_TIMING_E41']
    node=H.install(T)
    saved={n:getattr(Bank,n) for n in ('prefetch','acquire','finish')}
    silent_span=P.copy_span
    state={'cuda_events':False}

    def prefetch(self,group,slot):
        if self.loaded[slot]==group:return saved['prefetch'](self,group,slot)
        with torch.cuda.stream(self.stream):
            if self.done[slot] is not None:self.stream.wait_event(self.done[slot])
            with real_copy_span(f'backbone/group={group}',self.device,self.stream,self.groups[group].host.numel()):
                return saved['prefetch'](self,group,slot)

    def acquire(self,group,prefetch_next=True):
        result=saved['acquire'](self,group,prefetch_next)
        span=real_copy_span(f'backbone/compute_group={group}',self.device,torch.cuda.current_stream(self.device),0)
        span.__enter__();self._e41_span=span
        return result

    def finish(self,slot):
        span=getattr(self,'_e41_span',None)
        if span is not None:span.__exit__(None,None,None);self._e41_span=None
        return saved['finish'](self,slot)

    @PromptServer.instance.routes.post('/h3multistream/e41/events')
    async def events(request):
        running,pending=PromptServer.instance.prompt_queue.get_current_queue()
        if running or pending:return web.json_response({'error':'queue must be idle'},status=409)
        enable=bool((await request.json())['enabled'])
        if P._path is not None:P.flush()
        P.copy_span=real_copy_span if enable else silent_span
        VS.copy_span=P.copy_span
        for name,fn in dict(prefetch=prefetch,acquire=acquire,finish=finish).items():
            setattr(Bank,name,fn if enable else saved[name])
        state['cuda_events']=enable
        return web.json_response(state)
    return node
