# SPDX-License-Identifier: GPL-3.0-only
"""Resident H3 tensor parallelism. Dense INT8/float weights, two equal ranks.

Weights live for one sampling execution, and are released by ON_CLEANUP.
All large matrices are output-row sharded. Gather input features before each
projection to preserve the checkpoint's full-row INT8 activation scales.
"""
import threading
import time
import contextlib

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.quant_ops as Q
from comfy.ldm.minimax import model as H
from comfy.ldm.modules import attention as A
from comfy.quant_ops import QuantizedTensor

from . import split as S
from .cast import staged_size
from .log import log


def tp_attention(q, k, v, heads, backend, preferred, options, output_dtype=None):
    if backend == 'pytorch_fp32':
        return A.optimized_attention(q, k, v, heads, preferred_attention=preferred,
                                     mask=None, skip_reshape=True, transformer_options=options).squeeze(0)
    q, k, v = q.take(), k.take(), v.take()
    dtype = q.dtype
    if backend in ('sparse_kitchen_int8', 'sparse_kitchen_sage_int8'):
        out = options['h3_tp_sparse_attention'](q, k, v, options)
    elif backend == 'flash_turing_fp16':
        import flash_attn_turing
        q, k, v = [t.transpose(1, 2).to(torch.float16).contiguous() for t in (q, k, v)]
        out, _ = flash_attn_turing.fwd(q, k, v, q.shape[-1] ** -0.5, False)
    elif backend == 'sage_turing_int8':
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        q, k, v = [t.transpose(1, 2).to(torch.float16).contiguous() for t in (q, k, v)]
        out = sageattn_qk_int8_pv_fp16_cuda(
            q, k, v, tensor_layout='NHD', is_causal=False,
            sm_scale=q.shape[-1] ** -0.5, pv_accum_dtype='fp32')
    elif backend == 'pytorch_fp16':
        q, k, v = [t.to(torch.float16) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    else:
        raise ValueError(f'Unknown H3 TP attention backend: {backend}')
    return out.reshape(out.shape[1], -1).to(dtype if output_dtype is None else output_dtype)


def row_indices(start, stop, width, repeats=1):
    return torch.cat([torch.arange(start, stop) + i * width for i in range(repeats)])


class LinearShard:
    def __init__(self, linear, device, rows=None, columns=None, bias=True, lora=None):
        for name in ('weight', 'bias'):
            if getattr(linear, name + '_function', []) or getattr(linear, name + '_lowvram_function', None):
                raise ValueError('H3 TP does not yet support LoRA or parameter patches')
        w = linear.weight
        self.quantized = isinstance(w, QuantizedTensor)
        self.scale = None
        self.convrot = False
        self.groupsize = 256
        if self.quantized:
            if w._layout_cls != 'TensorWiseINT8Layout' or w._params.transposed:
                raise ValueError('H3 TP supports non-transposed TensorWiseINT8Layout only')
            w, scale = Q.TensorWiseINT8Layout.get_plain_tensors(w)
            self.convrot = linear.weight._params.convrot
            self.groupsize = linear.weight._params.convrot_groupsize
            if rows is not None and scale.numel() != 1:
                scale = scale.index_select(0, rows.to(scale.device))
            self.scale = scale.detach().to(device, copy=True)
        if columns is not None:
            lo, hi = columns
            if self.convrot and (lo % self.groupsize or hi % self.groupsize):
                raise ValueError('TP column boundaries must align with ConvRot groups')
            w = w[:, lo:hi]
        if rows is not None:
            w = w.index_select(0, rows.to(w.device))
        # clone avoids keeping a full checkpoint/storage alive behind a small view.
        self.weight = w.detach().to(device, copy=not (rows is not None and w.device.type == 'cpu' and torch.device(device).type == 'cpu')).contiguous()
        b = linear.bias if bias else None
        if b is not None and rows is not None:
            b = b.index_select(0, rows.to(b.device))
        self.bias = None if b is None else b.detach().to(device, copy=True)
        self.lora_down = self.lora_up = None
        self.lora_strength = 0.0
        self.lora_precision = 'fp16'
        if lora is not None:
            down, up, self.lora_strength, self.lora_precision = lora
            if columns is not None:
                down = down[:, columns[0]:columns[1]]
            if rows is not None:
                up = up.index_select(0, rows.to(up.device))
            dtype = torch.float16 if self.lora_precision == 'fp16' else torch.float32
            self.lora_down = down.detach().to(device=device, dtype=dtype, copy=True).contiguous()
            self.lora_up = up.detach().to(device=device, dtype=dtype, copy=True).contiguous()


    def __call__(self, x, input_act=None):
        if self.quantized:
            out = Q.ck.int8_linear(x.contiguous(), self.weight, self.scale,
                                  None if self.bias is None else self.bias.to(x.dtype), x.dtype,
                                  convrot=self.convrot, convrot_groupsize=self.groupsize,
                                  input_act=input_act)
        else:
            source = x
            if input_act == 'swiglu':
                gate, value = x.chunk(2, dim=-1)
                source = F.silu(gate) * value
            out = F.linear(source, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        if self.lora_down is not None and self.lora_strength != 0:
            from .tp_lora import add_lora_
            if input_act == 'swiglu':
                gate, value = x.chunk(2, dim=-1)
                x = F.silu(gate) * value
            out = add_lora_(out, x, self.lora_down, self.lora_up, self.lora_strength, self.lora_precision)
        return out

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.weight, self.scale, self.bias, self.lora_down, self.lora_up) if t is not None)


class BlockShard:
    def __init__(self, block, rank, device, lora=None):
        lora = lora or {}
        heads, hd = block.attn.heads, block.attn.head_dim
        ffn = block.mlp.fc2.weight.shape[1]
        adaln = block.adaln_proj
        if heads % 2 or ffn % 2 or adaln.linear.weight.shape[0] % 2:
            raise ValueError('H3 TP=2 requires even head, FFN and AdaLN output counts')
        self.heads, self.hd = heads // 2, hd
        hidden = block.attn.out_proj.weight.shape[0]
        if hidden % 2:
            raise ValueError('H3 TP=2 requires even hidden width')
        a, b = rank * self.heads * hd, (rank + 1) * self.heads * hd
        f0, f1 = rank * (ffn // 2), (rank + 1) * (ffn // 2)
        m = adaln.linear.weight.shape[0] // 2
        self.qkv = LinearShard(block.attn.qkv_proj, device, row_indices(a, b, heads * hd, 3), lora=lora.get('attn.qkv_proj'))
        hidden_rows = torch.arange(rank * (hidden // 2), (rank + 1) * (hidden // 2))
        self.out = LinearShard(block.attn.out_proj, device, hidden_rows, lora=lora.get('attn.out_proj'))
        self.fc1 = LinearShard(block.mlp.fc1, device, row_indices(f0, f1, ffn, 2), lora=lora.get('mlp.fc1'))
        self.fc2 = LinearShard(block.mlp.fc2, device, hidden_rows, lora=lora.get('mlp.fc2'))
        from .swiglu_fusion import maybe_wrap as wrap_swiglu
        from . import tp_lora
        from .internal_precision import wrap as wrap_precision
        wrap_precision(self, Q, tp_lora)
        self.fc2 = wrap_swiglu(self.fc2, Q, tp_lora)
        from .fc2_gemm import maybe_wrap
        self.fc2 = maybe_wrap(self.fc2)
        self.adaln = LinearShard(adaln.linear, device, torch.arange(rank * m, (rank + 1) * m), lora=lora.get('adaln_proj.linear'))
        self.mod_shape = (adaln.modalities, adaln.expand, adaln.hidden, adaln.apply_silu)
        self.n1 = block.norm1.weight.detach().to(device).clone()
        self.n2 = block.norm2.weight.detach().to(device).clone()
        self.qn = block.attn.q_norm.weight.detach().to(device).clone()
        self.kn = block.attn.k_norm.weight.detach().to(device).clone()
        self.eps1, self.eps2 = block.norm1.eps, block.norm2.eps
        self.qeps = block.attn.q_norm.eps
        self.preferred_attention = block.attn.comfy_attention
        self.attention_backend = 'pytorch_fp32'
        self.communication = 'legacy'
        self.chunk_tokens = 2048
        self.sparse_attention = None
        self.stage = contextlib.nullcontext

    def nbytes(self):
        return sum(x.nbytes() for x in (self.qkv, self.out, self.fc1, self.fc2, self.adaln)) + sum(
            t.numel() * t.element_size() for t in (self.n1, self.n2, self.qn, self.kn))

    def forward(self, rank, x, emb, segments, rope, options, exchange):
        if self.attention_backend in ('sparse_kitchen_int8', 'sparse_kitchen_sage_int8'):
            if self.sparse_attention is None:
                raise RuntimeError('Sparse Kitchen rank was not initialized')
            options = dict(options, h3_tp_sparse_attention=self.sparse_attention,
                           h3_tp_sparse_stage=self.stage)
        if self.communication != 'legacy':
            return self.forward_peer(rank, x, emb, segments, rope, options, exchange)
        modalities, expand, hidden, silu = self.mod_shape
        mod = self.adaln(F.silu(emb) if silu else emb)
        mod = exchange.all_gather(rank, mod.T.contiguous(), x.device).T.contiguous()
        sm, cm, gm, sf, cf, gf = mod.view(emb.shape[0] * modalities, expand * hidden).chunk(expand, -1)
        h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n1.to(x.dtype), self.eps1), sm, cm, segments)
        qkv = self.qkv(h)
        del h
        q, k, v = qkv.split(self.heads * self.hd, -1)
        count = x.shape[0]
        q = q.view(1, count, self.heads, self.hd)
        k = k.view(1, count, self.heads, self.hd)
        Q.ck.rms_rope_split_half_(q, k, rope, self.qn, self.kn, epsilon=self.qeps, rot_dim=rope.shape[-3] * 2)
        q = A.AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
        k = A.AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
        v = A.AttentionTensorContainer(v.view(count, self.heads, self.hd).transpose(0, 1).unsqueeze(0))
        attn = tp_attention(q, k, v, self.heads, self.attention_backend, self.preferred_attention, options)
        del q, k, v, qkv
        attended = torch.empty_like(x)
        for start in range(0, count, 1024):
            stop = min(start + 1024, count)
            features = gather_features(exchange, rank, attn[start:stop])
            attended[start:stop] = gather_features(exchange, rank, self.out(features))
        del features
        del attn
        x = H._mod_gate(x, gm, attended, segments)
        del attended
        h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n2.to(x.dtype), self.eps2), sf, cf, segments)
        output = torch.empty_like(x)
        for start in range(0, count, 1024):
            stop = min(start + 1024, count)
            # gathered order is gate0, up0, gate1, up1; SwiGLU needs gate0, gate1, up0, up1.
            features = gather_features(exchange, rank, self.fc1(h[start:stop]))
            features = features.reshape(stop - start, 2, 2, -1).transpose(1, 2).reshape(stop - start, -1)
            output[start:stop] = gather_features(exchange, rank, self.fc2(features, input_act='swiglu'))
        del features
        del h
        return H._mod_gate(x, gf, output, segments)

    def forward_peer(self, rank, x, emb, segments, rope, options, ex):
        modalities, expand, hidden, silu = self.mod_shape
        with self.stage('adaln'):
            mod = self.adaln(F.silu(emb) if silu else emb)
        with self.stage('adaln_exchange'):
            mod = ex.gather(rank, mod)
        with self.stage('norm1_modulation'):
            sm, cm, gm, sf, cf, gf = mod.view(emb.shape[0] * modalities, expand * hidden).chunk(expand, -1)
            h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n1.to(x.dtype), self.eps1), sm, cm, segments)
        with self.stage('qkv'):
            qkv = self.qkv(h)
            del h
        with self.stage('qk_norm_rope'):
            q, k, v = qkv.split(self.heads * self.hd, -1)
            count = x.shape[0]
            q = q.view(1, count, self.heads, self.hd)
            k = k.view(1, count, self.heads, self.hd)
            Q.ck.rms_rope_split_half_(q, k, rope, self.qn, self.kn, epsilon=self.qeps, rot_dim=rope.shape[-3] * 2)
            q = A.AttentionTensorContainer(q[0].transpose(0, 1).unsqueeze(0))
            k = A.AttentionTensorContainer(k[0].transpose(0, 1).unsqueeze(0))
            v = A.AttentionTensorContainer(v.view(count, self.heads, self.hd).transpose(0, 1).unsqueeze(0))
        with self.stage('attention'):
            attn = tp_attention(q, k, v, self.heads, self.attention_backend, self.preferred_attention, options,
                                output_dtype=torch.float16)
            del q, k, v, qkv
        with self.stage('attn_input_exchange'):
            features = ex.gather(rank, attn).to(x.dtype)
            del attn
        projected = x.new_empty((count, hidden // 2))
        with self.stage('attn_out_projection'):
            for start in range(0, count, self.chunk_tokens):
                stop = min(start + self.chunk_tokens, count)
                projected[start:stop] = self.out(features[start:stop])
            del features
        with self.stage('attn_output_exchange'):
            attended = ex.gather(rank, projected)
            del projected
        with self.stage('norm2_modulation'):
            x = H._mod_gate(x, gm, attended, segments)
            del attended
            h = H._mod_scale_shift(F.rms_norm(x, (hidden,), self.n2.to(x.dtype), self.eps2), sf, cf, segments)
        projected = x.new_empty((count, hidden // 2))
        with self.stage('mlp_fc1'):
            first = self.fc1(h[:self.chunk_tokens])
        with self.stage('mlp_input_exchange'):
            pending = ex.start(rank, first)
            del first
        for start in range(0, count, self.chunk_tokens):
            stop = min(start + self.chunk_tokens, count)
            if stop < count:
                with self.stage('mlp_fc1'):
                    following = self.fc1(h[stop:stop + self.chunk_tokens])
                with self.stage('mlp_input_exchange'):
                    next_pending = ex.start(rank, following)
                    del following
            with self.stage('mlp_input_exchange'):
                features = ex.finish(pending, swiglu=True)
                del pending
            with self.stage('mlp_fc2_swiglu'):
                projected[start:stop] = self.fc2(features, input_act='swiglu')
                del features
            if stop < count:
                pending = next_pending
                del next_pending
        del h
        with self.stage('mlp_output_exchange'):
            output = ex.gather(rank, projected)
            del projected
        with self.stage('residual'):
            result = H._mod_gate(x, gf, output, segments)
        # Bound cross-stream allocator lifetimes to one block; token chunks
        # inside it still overlap copies and GEMMs on independent streams.
        with self.stage('block_complete'):
            torch.cuda.current_stream(x.device).synchronize()
        return result


def gather_features(exchange, rank, value):
    return exchange.all_gather(rank, value.T.contiguous(), value.device).T.contiguous()


class ResidentTP:
    def __init__(self, reserve_gib=4.0, attention_backend='pytorch_fp32', communication='legacy', chunk_tokens=2048,
                 weight_residency='full', sparse_video_keep=0.5, packed_cache=False, cache_generation=0, num_groups=3, resident_slots=2, residency_safety_gib=2.0, cpu_pack_layout="canonical"):
        if attention_backend not in ('pytorch_fp32', 'pytorch_fp16', 'flash_turing_fp16', 'sage_turing_int8', 'sparse_kitchen_int8', 'sparse_kitchen_sage_int8'):
            raise ValueError(f'Unknown H3 TP attention backend: {attention_backend}')
        self.attention_backend = attention_backend
        self.sparse_video_keep = float(sparse_video_keep)
        if not 0 < self.sparse_video_keep <= 1:
            raise ValueError('sparse_video_keep must be in (0, 1]')
        self.sparse_ranks = []
        if communication not in ('legacy', 'event_p2p'):
            raise ValueError(f'Unknown TP communication mode: {communication}')
        self.communication = communication
        self.chunk_tokens = int(chunk_tokens)
        self.reserve_gib = reserve_gib
        self.ranks = None
        self.owner = None
        if weight_residency not in ('full', 'three_groups', 'manual_groups', 'auto_tokens'):
            raise ValueError(f'Unknown weight residency: {weight_residency}')
        self.weight_residency = weight_residency
        self.num_groups = int(num_groups)
        self.resident_slots = int(resident_slots)
        self.residency_safety_gib = float(residency_safety_gib)
        if not 1.0 <= self.residency_safety_gib <= 8.0:
            raise ValueError('residency_safety_gib must be in [1, 8]')
        self.tokens = None
        self.banks = []
        self.donated = False
        self.packed_cache = bool(packed_cache)
        self.cache_generation = int(cache_generation)
        if cpu_pack_layout not in ("canonical", "grouped"): raise ValueError("Unknown CPU pack layout")
        self.cpu_pack_layout = cpu_pack_layout

    def release(self, *_):
        if self.ranks is not None:
            for device in self.devices:
                torch.cuda.synchronize(device)
            for bank in self.banks:
                bank.close()
            self.banks = []
            self.sparse_ranks = []
            self.ranks = None
            self.owner = None
            self.donated = False
            log.info('[H3 TP] resident shards released after sampling')

    def prepare(self, model, devices, lora_config=None):
        if self.ranks is not None:
            if self.owner is not model or self.devices != devices:
                raise ValueError('H3 TP model/device changed during one sampling execution')
            return
        if len(devices) != 2 or len(set(devices)) != 2:
            raise ValueError('H3 TP currently requires exactly two distinct CUDA GPUs')
        if self.attention_backend in ('sparse_kitchen_int8', 'sparse_kitchen_sage_int8'):
            from .tp_sparse import preflight, SparseAttention
            if self.attention_backend == 'sparse_kitchen_sage_int8':
                from .tp_kitchen_sage import KitchenSageAttention as SparseAttention
            preflight(devices)
            self.sparse_ranks = [SparseAttention(self.sparse_video_keep) for _ in devices]
            log.info('[H3 TP] Sparse Kitchen fixed top-k keep=%s; 64Qx32KV; raster; conditioning dense', self.sparse_video_keep)
        if self.attention_backend == 'flash_turing_fp16':
            import flash_attn_turing
            for device in devices:
                if torch.cuda.get_device_capability(device) != (7, 5):
                    raise ValueError('flash_turing_fp16 requires a Turing SM75 GPU')
        if self.attention_backend == 'sage_turing_int8':
            from sageattention import sageattn_qk_int8_pv_fp16_cuda
            for device in devices:
                if torch.cuda.get_device_capability(device) != (7, 5):
                    raise ValueError('sage_turing_int8 requires a Turing SM75 GPU and the SM75 Sage build')
        # Legacy model management may have preloaded complete blocks onto GPU0.
        # Return those originals to RAM before making the two resident shards.
        # QuantizedTensor._apply can rewrap already-CPU parameters and change
        # their identity. Do not perform a redundant transport on warm masters.
        if any(t.device.type != 'cpu' for t in list(model.blocks.parameters()) + list(model.blocks.buffers())):
            model.blocks.to('cpu')
        for module in model.blocks.modules():
            if hasattr(module, 'comfy_cast_weights') and not module.comfy_cast_weights:
                if not hasattr(module, 'prev_comfy_cast_weights'):
                    module.prev_comfy_cast_weights = module.comfy_cast_weights
                module.comfy_cast_weights = True
        S.free_rank_devices(devices, devices[0], model)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        def block_lora(i):
            if not lora_config:
                return None
            prefix = f'blocks.{i}.'
            return {name[len(prefix):]: (*pair, lora_config['strength'], lora_config['precision'])
                    for name, pair in lora_config['pairs'].items() if name.startswith(prefix)}
        self.devices = list(devices)
        self.owner = model
        self.ranks = [[], []]
        try:
            if self.weight_residency != 'full':
                from .tp_residency import PackedGroup, TwoSlotBank
                from .residency_plan import ranges, choose
                count = len(model.blocks)
                groups, slots = (3, 2) if self.weight_residency == 'three_groups' else (self.num_groups, self.resident_slots)
                if self.weight_residency == 'auto_tokens':
                    import os
                    if (count != 50 or self.attention_backend != 'sparse_kitchen_int8'
                        or self.sparse_video_keep != 0.5 or self.chunk_tokens > 4096
                        or os.environ.get('H3_TP_INTERNAL_OUTPUT', 'original') != 'original'):
                        raise ValueError('auto_tokens profile requires H3 50 blocks, Kitchen50, chunk<=4096, original internal output')
                    estimates=[]
                    for rank in range(2):
                        probe=BlockShard(model.blocks[0],rank,torch.device('cpu'),block_lora(0))
                        if not (probe.qkv.quantized and probe.fc1.quantized):
                            raise ValueError('auto_tokens empirical profile currently requires INT8 backbone; use manual_groups for other weights')
                        estimates.append(probe.nbytes()+65536)
                        del probe
                    self.residency_plan=choose(count,estimates,self.tokens,
                        [mm.get_free_memory(d) for d in devices],self.reserve_gib,self.residency_safety_gib,max_slots=self.resident_slots)
                    groups,slots=self.residency_plan['groups'],self.residency_plan['slots']
                    log.info('[H3 Residency] selected %s',{k:v for k,v in self.residency_plan.items() if k != 'candidates'})
                if not 2 <= slots < groups: raise ValueError('Need 2 <= resident_slots < num_groups')
                self.group_ranges = ranges(count, groups)
                self.active_slots = slots
                from . import packed_cache as pc
                from .profile_e35 import scope
                canonical = self.cpu_pack_layout == 'canonical'
                layout_key = ('canonical-v1',) if canonical else ('grouped-v1', tuple(self.group_ranges))
                key = pc.signature(model, lora_config, self.cache_generation) + (layout_key,) if self.packed_cache else None
                payload = pc.get(model, key) if self.packed_cache else None
                hit = payload is not None
                with scope('backbone/cpu_pack', cache_hit=hit, layout=self.cpu_pack_layout):
                    if not hit:
                        if self.packed_cache:pc.clear()
                        if canonical:
                            from .canonical_pack import CanonicalPack
                            payload = [CanonicalPack(count, lambda i,r=rank: BlockShard(model.blocks[i], r, torch.device('cpu'), block_lora(i))) for rank in range(2)]
                        else:
                            payload=[]
                            for rank in range(2):
                                packed=[]
                                for start,stop in self.group_ranges:
                                    blocks=[BlockShard(model.blocks[i],rank,torch.device('cpu'),block_lora(i)) for i in range(start,stop)]
                                    packed.append(PackedGroup(blocks));del blocks
                                payload.append(packed)
                        if self.packed_cache:pc.put(model,key,payload)
                with scope('backbone/group_views', layout=self.cpu_pack_layout):
                    all_groups=[master.groups(self.group_ranges) for master in payload] if canonical else payload
                # Rebind runtime options on every use; cached templates hold no
                # per-execution CUDA buffers or stream state.
                for rank, (device, groups) in enumerate(zip(devices, all_groups)):
                    for group in groups:
                        for block in group.templates:
                            block.attention_backend = self.attention_backend
                            block.communication = self.communication
                            block.chunk_tokens = self.chunk_tokens
                            if self.sparse_ranks:
                                block.sparse_attention = self.sparse_ranks[rank]
                            elif hasattr(block, 'sparse_attention'):
                                del block.sparse_attention
                    capacity = max(group.host.numel() for group in groups)
                    workspace = self.reserve_gib
                    safety = 0.0
                    if self.weight_residency == 'auto_tokens':
                        workspace = self.residency_plan['workspace_gib']
                        safety = self.residency_safety_gib
                    if mm.get_free_memory(device) < self.active_slots * capacity + (workspace + safety) * 2**30:
                        raise RuntimeError(f'H3 TP rank {rank}: actual packed slots exceed workspace budget')
                    self.banks.append(TwoSlotBank(groups, device, self.active_slots))
                log.info('[H3 Load] backbone packed cache %s', 'HIT' if hit else 'MISS')
                log.info('[H3 TP] groups %s, resident slots per rank GiB: %s; backend=%s',
                         self.group_ranges, [sum(b.numel() for b in bank.buffers) / 2**30 for bank in self.banks],
                         self.attention_backend)
                return
            for i, block in enumerate(model.blocks):
                for rank, device in enumerate(devices):
                    free = mm.get_free_memory(device)
                    # Conservative per-block bound; no full FP32 weight expansion.
                    need = staged_size(block) // 2 + 16 * 2**20
                    if free < need + self.reserve_gib * 2**30:
                        raise RuntimeError(f'H3 TP rank {rank}: block {i} cannot fit with '
                                           f'{self.reserve_gib:g} GiB working-space reserve; refusing offload')
                    with torch.cuda.device(device):
                        self.ranks[rank].append(BlockShard(block, rank, device, block_lora(i)))
                        self.ranks[rank][-1].attention_backend = self.attention_backend
                        self.ranks[rank][-1].communication = self.communication
                        self.ranks[rank][-1].chunk_tokens = self.chunk_tokens
                        if self.sparse_ranks:
                            self.ranks[rank][-1].sparse_attention = self.sparse_ranks[rank]
            sizes = [sum(b.nbytes() for b in blocks) / 2**30 for blocks in self.ranks]
            log.info('[H3 TP] all %d blocks resident, weights GiB per rank: %s', len(model.blocks), sizes)
            log.info('[H3 TP] attention backend: %s; residual and modulation retain model dtype', self.attention_backend)
        except BaseException:
            self.release()
            raise

    def run(self, dm, args, user_patches, devices, shares=None, exchange='p2p', *unused):
        if user_patches or args['transformer_options'].get(S.OVERRIDE):
            raise ValueError('H3 TP currently supports dense attention without block/attention patches')
        if exchange != 'p2p':
            raise ValueError('H3 TP currently requires p2p exchange')
        if self.donated:
            raise RuntimeError('Sampler evaluated again after a declared final Euler step; disable VAE slot donation')
        tokens = args['img'].shape[0]
        if self.ranks is not None and self.tokens != tokens:
            raise ValueError('Packed token count changed within sampling; refusing stale residency plan')
        self.tokens = tokens
        self.prepare(dm, devices, args['transformer_options'].get('h3_tp_runtime_lora'))
        opts = args['transformer_options']
        stage_key = opts.get('h3_vae_stage_key')
        last_sigma = opts.get('h3_vae_last_sigma')
        sigma = opts.get('sigmas')
        final_step = bool(stage_key and last_sigma is not None and isinstance(sigma, torch.Tensor)
                          and sigma.numel() and torch.all(torch.abs(sigma.detach().float() - last_sigma) < 1e-6).item()
                          and len(opts.get('cond_or_uncond', [0])) == 1)

        if self.communication == 'event_p2p':
            from .tp_comm import PeerExchange
            ex = PeerExchange(devices)
        else:
            ex = S._Collective('p2p', 2, devices)
        h = args['img']
        if self.communication == 'event_p2p':
            xs = [h.clone(), ex.copy_to(h, 1)]
            embs = [ex.copy_to(args['t_emb'], r) for r in range(2)]
            ropes = [ex.copy_to(args['rope_freqs'], r) for r in range(2)]
            segments = [[(a, b, ex.copy_to(row, r)) for a, b, row in args['mod_segments']] for r in range(2)]
            input_ready = [torch.cuda.Event() for _ in devices]
            for ready, device in zip(input_ready, devices):
                ready.record(torch.cuda.current_stream(device))
        else:
            xs = [h.clone(), S.to_device(h, devices[1])]
            embs = [S.to_device(args['t_emb'], d) for d in devices]
            ropes = [S.to_device(args['rope_freqs'], d) for d in devices]
            segments = [S._local_segments(args['mod_segments'], 0, h.shape[0], d) for d in devices]
        errors = []
        inference, grad = torch.is_inference_mode_enabled(), torch.is_grad_enabled()
        started = time.perf_counter()

        def worker(rank):
            try:
                with torch.cuda.device(devices[rank]), torch.inference_mode(inference), torch.set_grad_enabled(grad):
                    if self.communication == 'event_p2p':
                        torch.cuda.current_stream(devices[rank]).wait_event(input_ready[rank])
                    if self.weight_residency != 'full':
                        bank = self.banks[rank]
                        bank.final_step = final_step
                        initial_stats = torch.cuda.memory_stats(devices[rank])
                        for group, (start, stop) in enumerate(self.group_ranges):
                            last_group = final_step and group == len(self.group_ranges) - 1
                            slot, blocks = bank.acquire(group, prefetch_next=not last_group)
                            if last_group:
                                from .vae_staging import donate
                                if donate(stage_key, rank, bank, slot):
                                    self.donated = True
                            from .profile_e35 import copy_span
                            import contextlib
                            event_scope = copy_span('backbone/final_group_execute', devices[rank], torch.cuda.current_stream(devices[rank]), 0) if last_group else contextlib.nullcontext()
                            with event_scope:
                                for i, block in zip(range(start, stop), blocks):
                                    mm.throw_exception_if_processing_interrupted()
                                    options = dict(args['transformer_options'], block_index=i)
                                    from .tp_diagnostics import install_stage
                                    install_stage(block, rank, i)
                                    xs[rank] = block.forward(rank, xs[rank], embs[rank], segments[rank], ropes[rank], options, ex)
                            bank.finish(slot)
                            if self.weight_residency == 'auto_tokens':
                                stats = torch.cuda.memory_stats(devices[rank])
                                if any(stats.get(k, 0) > initial_stats.get(k, 0) for k in ('num_alloc_retries', 'num_ooms')):
                                    raise RuntimeError('H3_RESIDENCY_PRESSURE: allocator retry/OOM; stop instead of silently slowing down')
                    else:
                        for i, block in enumerate(self.ranks[rank]):
                            mm.throw_exception_if_processing_interrupted()
                            options = dict(args['transformer_options'], block_index=i)
                            xs[rank] = block.forward(rank, xs[rank], embs[rank], segments[rank], ropes[rank], options, ex)
                    torch.cuda.synchronize(devices[rank])
            except BaseException as error:
                log.exception('[H3 TP] primary worker failure rank=%d', rank)
                errors.append(error)
                ex.abort()

        threads = [threading.Thread(target=worker, args=(r,), name=f'h3-tp-{r}') for r in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.communication == 'event_p2p':
            try:
                ex.close()
            except BaseException as cleanup_error:
                if not errors: raise
                log.error('[H3 TP] secondary exchange cleanup failure: %r', cleanup_error)
        if errors:
            raise next((e for e in errors if not isinstance(e, threading.BrokenBarrierError)), errors[0])
        log.info('[H3 TP] %d tokens, %d blocks, %.2fs; residency=%s',
                 h.shape[0], len(dm.blocks), time.perf_counter() - started, self.weight_residency)
        return xs[0]
