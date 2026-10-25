# SPDX-License-Identifier: GPL-3.0-only
"""Explicit runtime Larry LoRA for full H3. Never merges into INT8 weights."""
import math
import torch
import torch.nn.functional as F
import comfy.ops
import comfy.utils
import comfy.patcher_extension
import folder_paths
from .log import log

OPTION = 'h3_tp_runtime_lora'


def add_lora_(out, x, down, up, strength=1.0, precision='fp16', chunk=2048):
    """Bound temporary activations; TP output-row split needs no extra collective."""
    if strength == 0:
        return out
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    y = out.reshape(-1, out.shape[-1])
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        if precision == 'fp16' and x.is_cuda:
            low = torch.mm(x[start:stop].to(torch.float16), down.T, out_dtype=torch.float32)
            delta = torch.mm(low.to(torch.float16), up.T, out_dtype=torch.float32)
        else:
            delta = F.linear(F.linear(x[start:stop].float(), down.float()), up.float())
        y[start:stop].add_(delta.to(y.dtype), alpha=strength)
    return out


def load_pairs(path, dm):
    state = comfy.utils.load_torch_file(path, safe_load=True)
    names = sorted({k[:-len('.lora_A.weight')] for k in state if k.endswith('.lora_A.weight')})
    expected = {n+s for n in names for s in ('.lora_A.weight', '.lora_B.weight')}
    if not names or set(state) != expected:
        raise ValueError('Expected plain Larry lora_A/lora_B weights; no unknown tensors may be dropped')
    pairs = {}
    for name in names:
        module = comfy.utils.get_attr(dm, name)
        down, up = state[name+'.lora_A.weight'], state[name+'.lora_B.weight']
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            raise ValueError(f'Invalid LoRA rank: {name}')
        if tuple(module.weight.shape) != (up.shape[0], down.shape[1]):
            raise ValueError(f'LoRA/base shape mismatch: {name}')
        if not torch.isfinite(down).all() or not torch.isfinite(up).all():
            raise ValueError(f'Nonfinite LoRA: {name}')
        pairs[name] = (down, up)
    return pairs


class H3TPLarryLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'lora_name': (folder_paths.get_filename_list('loras'),),
            'enabled': ('BOOLEAN', {'default': True}),
            'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.05}),
            'compute_precision': (['fp32', 'fp16'], {'default': 'fp32'})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'advanced/model'
    DESCRIPTION = ('Larry plain A/B runtime LoRA for full H3 + H3 TP. Preserves INT8 base; '
                   'includes all AdaLN/refiner/output updates. No pruned bases or stacked adapters. '
                   'Use before H3 Tensor Parallel. FP32 is the validated default. '
                   'FP16 is experimental: audio NaNs observed in an 8-step run.')

    def apply(self, model, lora_name, enabled=True, strength=1.0, compute_precision='fp32'):
        if not enabled or strength == 0:
            return (model.clone(),)
        if compute_precision not in ('fp16', 'fp32') or not math.isfinite(strength):
            raise ValueError('Invalid LoRA precision or strength')
        dm = model.model.diffusion_model
        if getattr(dm, 'use_adaln_curves', False):
            raise ValueError('This TP LoRA adapter requires the full AdaLN base, not pruned/curve weights')
        if model.patches or model.model_options.get('transformer_options', {}).get(OPTION):
            raise ValueError('Stacked/merged adapters are not supported; use one explicit runtime LoRA')
        pairs = load_pairs(folder_paths.get_full_path_or_raise('loras', lora_name), dm)
        block_pairs = {n: p for n,p in pairs.items() if n.startswith('blocks.')}
        for name in block_pairs:
            parts=name.split('.',2)
            if parts[2] not in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2','adaln_proj.linear'):
                raise ValueError(f'Unsupported TP adapter target: {name}')
        m = model.clone()
        m.model_options.setdefault('transformer_options', {})[OPTION] = {
            'pairs': block_pairs, 'strength': strength, 'precision': compute_precision, 'name': lora_name}
        def require_tp(executor, x, timestep, context, transformer_options={}, *args, **kwargs):
            if not transformer_options.get('h3_tp_runtime_enabled'):
                raise ValueError('H3 TP Larry LoRA must feed H3 Tensor Parallel; ordinary sampling would omit block adapters')
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, OPTION, require_tp)
        caches = []

        def delta(pair):
            cache = {}; caches.append(cache)
            def apply(out, x):
                key = (x.device, compute_precision)
                if key not in cache:
                    dtype = torch.float16 if compute_precision == 'fp16' and x.is_cuda else torch.float32
                    cache[key] = tuple(t.to(device=x.device, dtype=dtype).contiguous() for t in pair)
                return add_lora_(out, x, *cache[key], strength, compute_precision)
            return apply

        for name, pair in pairs.items():
            if name in block_pairs:
                continue
            module = comfy.utils.get_attr(dm, name)
            if name.endswith('.mlp.fc2'):
                # Fused INT8 SwiGLU bypasses fc2.forward. Patch the MLP once, keep
                # its original fused base path, and add the delta after SwiGLU.
                parent_name = name.rsplit('.',1)[0]
                parent = comfy.utils.get_attr(dm, parent_name)
                apply_delta = delta(pair)
                def mlp_forward(x, base=parent, add=apply_delta):
                    h = base.fc1(x)
                    out = comfy.ops.linear_input_act(base.fc2, h, 'swiglu')
                    gate, value = h.chunk(2, dim=-1)
                    return add(out, F.silu(gate)*value)
                m.add_object_patch('diffusion_model.'+parent_name+'.forward', mlp_forward)
            else:
                original = module.forward
                apply_delta = delta(pair)
                def forward(x, *args, original=original, add=apply_delta, **kwargs):
                    return add(original(x, *args, **kwargs), x)
                m.add_object_patch('diffusion_model.'+name+'.forward', forward)

        def cleanup(*args, **kwargs):
            for cache in caches:
                cache.clear()
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, OPTION, cleanup)
        log.info('[H3 TP LoRA] %s: %d TP targets + %d unsharded targets; strength=%g precision=%s',
                 lora_name, len(block_pairs), len(pairs)-len(block_pairs), strength, compute_precision)
        return (m,)
