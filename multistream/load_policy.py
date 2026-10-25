"""Avoid registering original H3 block pages that the TP packed bank replaces."""
import functools
from .profile_e35 import scope

def should_skip(patcher,key):
    opts=patcher.model_options.get('transformer_options',{})
    return (opts.get('h3_tp_runtime_enabled',False) and opts.get('h3_tp_owned_pinning',False)
            and key.startswith('diffusion_model.blocks.') and not patcher.patches)

def install():
    from comfy.model_patcher import ModelPatcher
    original=ModelPatcher.pin_weight_to_device
    if getattr(original,'_h3_owned_pin_policy',False):return
    @functools.wraps(original)
    def pin(self,key):
        if should_skip(self,key):
            with scope('memory/tp_skip_original_pin',weight_key=key):return None
        return original(self,key)
    pin._h3_owned_pin_policy=True
    ModelPatcher.pin_weight_to_device=pin
