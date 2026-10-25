# SPDX-License-Identifier: GPL-3.0-only
"""Kitchen tile selection + independently built Sage SM75 sparse consumer."""
import contextlib
import os
from pathlib import Path
import threading
import torch
from .tp_sparse import SparseAttention

_extension = None
_lock = threading.Lock()

def extension(build=False):
    global _extension
    with _lock:
        if _extension is None:
            import importlib.util
            binary = Path(__file__).with_name('kitchen_sage_bin') / 'h3_kitchen_sage_sm75_v1.so'
            if binary.exists() and not build:
                spec = importlib.util.spec_from_file_location('h3_kitchen_sage_sm75_v1', binary)
                _extension = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_extension)
                return _extension
            if not build:
                raise RuntimeError('Kitchen+Sage extension missing: run build_kitchen_sage.py in the H3 environment')
            from torch.utils.cpp_extension import load
            source = Path(__file__).with_name('kitchen_sage_src') / 'sparse_sage.cu'
            # Isolated module: never replaces the installed dense Sage package.
            _extension = load(name='h3_kitchen_sage_sm75_v1', sources=[str(source)],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_75,code=sm_75'],
                extra_cflags=['-O2'], verbose=False)
            import shutil
            binary.parent.mkdir(exist_ok=True)
            shutil.copy2(_extension.__file__, binary)
    return _extension

class KitchenSageAttention(SparseAttention):
    def __init__(self, keep):
        super().__init__(keep)
        self.consumer = extension()

    def __call__(self, q, k, v, options):
        packed = options.get('minimax_h3_layout')
        if packed is None:
            raise ValueError('Kitchen+Sage requires real minimax_h3_layout')
        layout = self.convert_layout(packed)
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[-1] != 128 or layout.seq_len != q.shape[-2]:
            raise ValueError('Kitchen+Sage requires matching HND QKV, D128 and matching layout')
        if torch.cuda.get_device_capability(q.device) != (7, 5):
            raise ValueError('Kitchen+Sage consumer requires SM75')
        stage = options.get('h3_tp_sparse_stage', contextlib.nullcontext)
        with stage('sparse_cast'):
            q, k, v = [x.to(torch.float16).contiguous() for x in (q,k,v)]
        with stage('sparse_route'):
            lut, counts, metadata = self.router.build_lut(q,k,layout,self.keep)
        self.last_metadata = vars(metadata)
        with stage('sparse_quantize'):
            from sageattention.triton.quant_per_thread import per_thread_int8
            km = k.mean(dim=2, keepdim=True)
            qi, qs, ki, ks = per_thread_int8(q,k,km,tensor_layout='HND',BLKQ=64,WARPQ=16,BLKK=32,WARPK=32)
        with stage('sparse_kernel'):
            out = self.consumer.forward(qi,ki,v,qs,ks,lut.contiguous(),counts.contiguous())
        return out.transpose(1,2)
