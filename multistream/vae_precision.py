"""Shared H3 VAE precision policies for ordinary and staged TP decoding."""
import torch

DEFAULT_MODE = 'tp2_fp16'
TP_MODES = ('tp2_fp16', 'tp2_precision', 'tp2_fc1_fp16', 'tp2_fp32', 'tp2_conservative')
MODES = TP_MODES + ('single_fp32',)
TOOLTIP = ('FP16 TP (default): FP16 weights/activations, FP32 projection partials and P2P reductions. '
           'precision: FP32 attention/FC1/boundaries, FP16 QKV/out/FC2. '
           'fc1_fp16: precision plus FP16 FC1 inputs/weights with FP32 output. '
           'fp32: FP32 TP baseline. conservative: legacy experimental FP16 attention. '
           'For staged loading, match precision_mode and cache_generation in H3LoadPipeline and H3TPVAEDecode.')

def model_dtype(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown H3 VAE precision mode: {mode}')
    return torch.float16 if mode == 'tp2_fp16' else torch.float32

def engine_kwargs(mode):
    if mode not in TP_MODES:
        raise ValueError(f'H3 VAE staging/TP requires a TP2 mode, got {mode}')
    if mode in ('tp2_fp16', 'tp2_fp32'):
        return {}
    return dict(balanced=True, conservative=mode != 'tp2_fc1_fp16',
                attention_fp32=mode != 'tp2_conservative')

def cast_shard_weights(shard, balanced=False, conservative=False):
    """Apply the same storage casts to ordinary GPU and packed CPU shards."""
    if balanced:
        for name in (('qw', 'ow', 'dw') if conservative else ('qw', 'ow', 'fw', 'dw')):
            setattr(shard, name, getattr(shard, name).half())
    return shard
