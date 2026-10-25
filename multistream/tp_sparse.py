# SPDX-License-Identifier: GPL-3.0-only
"""Explicit native Sparse Kitchen TP adapter; fixed top-k, raster token order.

Only target-video to target-video connections may be removed. Each TP rank
routes its own attention heads. No node hooks, fallback, or global model patch.
Dependency: Zironic/H3-Optimizations c6bc3cfc58c18c3a6220aeea09116e1eebd91669.
"""
import contextlib
import math
import os
from pathlib import Path
import sys
import threading
import torch

_lock = threading.Lock()
_modules = None

def modules():
    global _modules
    with _lock:
        if _modules is None:
            source = Path(os.environ.get('H3_SPARSE_KITCHEN_SOURCE', str(Path(__file__).resolve().parents[1] / 'third_party/H3-Optimizations')))
            if not (source / 'h3_optimizations/native/int8_attention.py').is_file():
                raise RuntimeError(f'Sparse Kitchen dependency missing: {source}')
            sys.path.insert(0, str(source))
            from h3_optimizations.native import int8_attention as kernel
            from h3_optimizations.attention.sparse.router import SparseTileRouter
            from h3_optimizations.runtime.layout import from_packed_layout
            _modules = kernel, SparseTileRouter, from_packed_layout
    return _modules

def preflight(devices):
    kernel, _, _ = modules()
    for device in devices:
        with torch.cuda.device(device):
            if not kernel.int8_attention_is_available(device):
                raise RuntimeError(f'Sparse Kitchen unavailable on {device}')
            if not kernel.sparse_geometry_is_available(64, 32, device):
                raise RuntimeError(f'Sparse Kitchen 64Qx32KV failed self-test on {device}')

class SparseAttention:
    def __init__(self, keep):
        self.keep = float(keep)
        if not math.isfinite(self.keep) or not 0 < self.keep <= 1:
            raise ValueError('sparse_video_keep must be in (0, 1]')
        self.kernel, router, self.convert_layout = modules()
        self.router = router(q_tile=64, kv_tile=32)
        self.last_metadata = None
        self.shared_kv = os.environ.get('H3_KITCHEN_SHARED_KV', '0') == '1'
        if self.shared_kv:
            from .kitchen_occupancy import run
            self.shared_kv_run = run

        # E48 is opt-in: baseline keeps the previously validated deployment.
        plan = os.environ.get('H3_KITCHEN_KERNEL_PLAN', 'baseline')
        plans = {'baseline': 0, 'offset': 1, 'defer_barrier': 2, 'offset_barrier': 3}
        if plan not in plans:
            raise ValueError(f'Unknown H3_KITCHEN_KERNEL_PLAN: {plan}')
        self.optimized_attention = None
        if plans[plan]:
            if not self.shared_kv:
                raise ValueError('H3_KITCHEN_KERNEL_PLAN requires H3_KITCHEN_SHARED_KV=1')
            from .kitchen_offsets_adapter import Attention
            self.optimized_attention = Attention(self)
            self.optimized_attention.mode = plans[plan]

    def __call__(self, q, k, v, options):
        if self.optimized_attention is not None:
            result = self.optimized_attention(q, k, v, options)
            self.last_metadata = self.optimized_attention.last_metadata
            return result
        packed = options.get('minimax_h3_layout')
        if packed is None:
            raise ValueError('Sparse Kitchen requires real minimax_h3_layout; refusing to guess conditioning spans')
        layout = self.convert_layout(packed)
        if layout.seq_len != q.shape[-2]:
            raise ValueError('Sparse Kitchen layout and QKV sequence mismatch')
        stage = options.get('h3_tp_sparse_stage', contextlib.nullcontext)
        with stage('sparse_cast'):
            q, k, v = [x.to(torch.float16).contiguous() for x in (q, k, v)]
        with stage('sparse_route'):
            lut, counts, metadata = self.router.build_lut(q, k, layout, self.keep)
        self.last_metadata = vars(metadata)
        with stage('sparse_quantize'):
            carrier = self.kernel.prequantize_int8_attention(q, k, v, cta_k=64)
        route = self.kernel.BlockSparseRoute(lut, counts, q_tile=64, kv_tile=32, encoding='delta')
        with stage('sparse_kernel'):
            result = (self.shared_kv_run(carrier, route, 1) if self.shared_kv else
                      self.kernel.block_sparse_int8_attention_from_prequantized(carrier, route))
        return result.transpose(1, 2)
