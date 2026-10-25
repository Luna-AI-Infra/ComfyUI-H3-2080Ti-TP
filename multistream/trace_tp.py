"""Opt-in Nsight annotations; no extra CUDA synchronization in timed regions."""
import contextlib
import functools
import itertools
import json
import os
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F

ROWS = []
TLS = threading.local()
IDS = itertools.count()
STEP = 0
LOADS = [0, 0]


@contextlib.contextmanager
def stage(phase, rank=None, block=None, payload=0):
    parent = getattr(TLS, 'scope', (STEP, -1, -1))
    scope = (STEP, parent[1] if rank is None else rank, parent[2] if block is None else block)
    TLS.scope = scope
    uid = next(IDS)
    tag = f'H3TP|{scope[0]}|{scope[1]}|{scope[2]}|{phase}|{uid}'
    t0, c0 = time.perf_counter_ns(), time.thread_time_ns()
    torch.cuda.nvtx.range_push(tag)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        ROWS.append(dict(id=uid, step=scope[0], rank=scope[1], block=scope[2], phase=phase,
                         start_ns=t0, end_ns=time.perf_counter_ns(), cpu_ns=time.thread_time_ns()-c0,
                         tid=threading.get_native_id(), payload_bytes=payload))
        TLS.scope = parent


def install(T):
    H, A, Q = T.H, T.A, T.Q
    gather = T.gather_features
    original_forward = T.BlockShard.forward

    def forward(self, rank, x, emb, segments, rope, options, exchange):
        block = options['block_index']
        if self.communication == 'event_p2p':
            self.stage = stage
            with stage('block', rank, block):
                return original_forward(self, rank, x, emb, segments, rope, options, exchange)
        with stage('block', rank, block):
            modalities, expand, hidden, silu = self.mod_shape
            with stage('adaln'):
                mod = self.adaln(F.silu(emb) if silu else emb)
            with stage('adaln_exchange'):
                mod = exchange.all_gather(rank, mod.T.contiguous(), x.device).T.contiguous()
            with stage('norm1_modulation'):
                sm, cm, gm, sf, cf, gf = mod.view(emb.shape[0] * modalities, expand * hidden).chunk(expand, -1)
                h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n1.to(x.dtype), self.eps1), sm, cm, segments)
            with stage('qkv'):
                qkv = self.qkv(h)
                del h
            with stage('qk_norm_rope'):
                q, k, v = qkv.split(self.heads * self.hd, -1)
                count = x.shape[0]
                q = q.view(1, count, self.heads, self.hd)
                k = k.view(1, count, self.heads, self.hd)
                Q.ck.rms_rope_split_half_(q, k, rope, self.qn, self.kn, epsilon=self.qeps, rot_dim=rope.shape[-3] * 2)
                q = A.AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
                k = A.AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
                v = A.AttentionTensorContainer(v.view(count, self.heads, self.hd).transpose(0, 1).unsqueeze(0))
            with stage('attention'):
                attn = T.tp_attention(q, k, v, self.heads, self.attention_backend, self.preferred_attention, options)
                del q, k, v, qkv
            attended = torch.empty_like(x)
            for start in range(0, count, 1024):
                stop = min(start + 1024, count)
                with stage('attn_input_exchange'):
                    features = gather(exchange, rank, attn[start:stop])
                with stage('attn_out_projection'):
                    projected = self.out(features)
                with stage('attn_output_exchange'):
                    attended[start:stop] = gather(exchange, rank, projected)
                del projected
            del features, attn
            with stage('norm2_modulation'):
                x = H._mod_gate(x, gm, attended, segments)
                del attended
                h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n2.to(x.dtype), self.eps2), sf, cf, segments)
            output = torch.empty_like(x)
            for start in range(0, count, 1024):
                stop = min(start + 1024, count)
                with stage('mlp_fc1'):
                    projected = self.fc1(h[start:stop])
                with stage('mlp_input_exchange'):
                    features = gather(exchange, rank, projected)
                del projected
                with stage('mlp_reorder'):
                    features = features.reshape(stop - start, 2, 2, -1).transpose(1, 2).reshape(stop - start, -1)
                with stage('mlp_fc2_swiglu'):
                    projected = self.fc2(features, input_act='swiglu')
                with stage('mlp_output_exchange'):
                    output[start:stop] = gather(exchange, rank, projected)
                del projected
            del features, h
            with stage('residual'):
                return H._mod_gate(x, gf, output, segments)

    original_init = T.BlockShard.__init__

    @functools.wraps(original_init)
    def init(self, block, rank, device):
        index = LOADS[rank]
        LOADS[rank] += 1
        with stage('load_weights', rank, index):
            original_init(self, block, rank, device)

    original_gather = T.S._Collective.all_gather

    @functools.wraps(original_gather)
    def all_gather(self, rank, tensor, device):
        with stage('collective', rank, payload=tensor.numel()*tensor.element_size()*(self.n-1)):
            return original_gather(self, rank, tensor, device)

    original_collective_init = T.S._Collective.__init__

    @functools.wraps(original_collective_init)
    def collective_init(self, *args, **kwargs):
        original_collective_init(self, *args, **kwargs)
        wait = self.barrier.wait

        def traced_wait(*args, **kwargs):
            with stage('barrier_wait'):
                return wait(*args, **kwargs)
        self.barrier.wait = traced_wait

    original_run = T.ResidentTP.run

    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        global STEP
        STEP += 1
        if STEP == 1:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            with stage('step', -1, -1):
                return original_run(self, *args, **kwargs)
        finally:
            if STEP == 2:
                for d in self.devices:
                    torch.cuda.synchronize(d)
                torch.cuda.cudart().cudaProfilerStop()
                Path(os.environ['H3TP_TRACE']).write_text(json.dumps(ROWS))

    T.BlockShard.forward = forward
    T.BlockShard.__init__ = init
    T.S._Collective.all_gather = all_gather
    T.S._Collective.__init__ = collective_init
    from .tp_comm import PeerExchange
    peer_init, peer_start = PeerExchange.__init__, PeerExchange.start

    def traced_peer_init(self, *args, **kwargs):
        peer_init(self, *args, **kwargs)
        wait = self.barrier.wait
        def traced_wait(*args, **kwargs):
            with stage('barrier_wait'):
                return wait(*args, **kwargs)
        self.barrier.wait = traced_wait

    def traced_peer_start(self, rank, value):
        with stage('collective', rank, payload=value.numel()*value.element_size()):
            return peer_start(self, rank, value)

    PeerExchange.__init__ = traced_peer_init
    PeerExchange.start = traced_peer_start
    T.ResidentTP.run = run
