"""E65: reference-level data parallel encoding; no precision or temporal changes."""
import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import comfy.model_management as mm

MODULES = {
    'MiniMaxH3VideoVAE': ('encoder', 'quant_conv'),
    'MiniMaxH3AudioVAE': ('encoder', 'pre_block', 'mean_proj'),
}

def pixels(vae, value):
    x = vae.vae_encode_crop_pixels(value).movedim(-1, 1)
    if vae.latent_dim == 3 and x.ndim < 5:
        x = x.unsqueeze(2) if vae.not_video else x.movedim(1, 0).unsqueeze(0)
    return vae.process_input(x).to(dtype=vae.vae_dtype, device='cpu')

def clone_encoder(vae, device):
    source = vae.first_stage_model
    result = copy.copy(source)
    result._modules = {name: getattr(source, name) for name in MODULES[type(source).__name__]}
    result._parameters = dict(source._parameters)
    result._buffers = dict(source._buffers)
    # Deep-copy structure once, but transfer tensor storage directly to its final
    # GPU allocation instead of first copying hundreds of MiB on the CPU.
    memo = {}
    for p in result.parameters():
        memo[id(p)] = torch.nn.Parameter(p.detach().to(device=device, dtype=vae.vae_dtype, copy=True), requires_grad=False)
    for b in result.buffers():
        memo[id(b)] = b.detach().to(device=device, dtype=vae.vae_dtype, copy=True)
    return copy.deepcopy(result, memo).eval()

@torch.inference_mode()
def encode_jobs(jobs, mode='auto', allow_unvalidated_video=False):
    """jobs=[(VAE, NHWC video frames or NLC audio)]; preserve request order."""
    if mode not in ('auto', 'single', 'dp2'):
        raise ValueError('Unknown reference encoding mode')
    started = time.perf_counter()
    stats = {'mode': mode, 'jobs': len(jobs), 'precision': 'unchanged', 'items': []}
    def serial(reason):
        stats.update(path='single', reason=reason)
        out = [vae.encode(value) for vae, value in jobs]
        stats['seconds'] = time.perf_counter() - started
        logging.info('[H3 Ref DP] %s', json.dumps(stats))
        return out
    if mode == 'single' or len(jobs) < 2:
        return serial('explicit single or fewer than two references')
    has_video = any(type(v.first_stage_model).__name__ == 'MiniMaxH3VideoVAE' and x.shape[0] > 1 and v.vae_dtype != torch.float32 for v, x in jobs)
    if has_video and not allow_unvalidated_video:
        if mode == 'dp2':
            raise ValueError('Low-precision video encoder DP has unresolved cross-GPU numerical differences. FP32 video DP is validated; use auto/single or explicitly enable allow_unvalidated_video for an ablation.')
        return serial('low-precision video DP numerical validation gate; FP32 video/images/audio DP supported')
    if torch.cuda.device_count() < 2:
        if mode == 'dp2':
            raise RuntimeError('Reference DP2 requires two CUDA devices')
        return serial('fewer than two CUDA devices')
    for vae, _ in jobs:
        if type(vae.first_stage_model).__name__ not in MODULES or vae.patcher.patches:
            if mode == 'dp2':
                raise ValueError('Reference DP supports unpatched H3 visual/audio encoders only')
            return serial('unsupported or patched VAE')
    # Global Comfy model management is invoked only on the calling thread.
    mm.unload_all_models()
    for rank in range(2):
        torch.cuda.synchronize(rank)
    mm.soft_empty_cache()
    prepared = [pixels(v, x) for v, x in jobs]
    plans = [[], []]
    costs = [0, 0]
    for i in sorted(range(len(jobs)), key=lambda i: prepared[i].numel(), reverse=True):
        rank = min(range(2), key=lambda r: costs[r])
        plans[rank].append(i)
        costs[rank] += prepared[i].numel()
    # Conservative preflight, not a proof of peak allocation. OOM is propagated;
    # never hide an OOM by retrying more work on a potentially damaged context.
    for rank, indices in enumerate(plans):
        unique = {id(jobs[i][0]): jobs[i][0] for i in indices}
        weight_bytes = 0
        for vae in unique.values():
            model = vae.first_stage_model
            for name in MODULES[type(model).__name__]:
                weight_bytes += sum(p.numel() * torch.empty((), dtype=vae.vae_dtype).element_size() for p in getattr(model, name).parameters())
        activation = max((jobs[i][0].memory_used_encode(prepared[i].shape, jobs[i][0].vae_dtype) for i in indices), default=0)
        free = torch.cuda.mem_get_info(rank)[0]
        need = weight_bytes + activation + 2 * 1024**3
        stats.setdefault('budgets', []).append({'rank': rank, 'free': free, 'estimated_need': need})
        if free < need:
            if mode == 'dp2':
                raise RuntimeError(f'Reference DP memory preflight rejected GPU{rank}: need {need}, free {free}')
            return serial('memory preflight; before any DP model allocation')
    results = [None] * len(jobs)
    def worker(rank):
        models = {}
        stream = None
        try:
            with torch.inference_mode(), torch.cuda.device(rank):
                stream = torch.cuda.Stream(device=rank)
                with torch.cuda.stream(stream):
                    for i in plans[rank]:
                        mm.throw_exception_if_processing_interrupted()
                        vae, _ = jobs[i]
                        key = id(vae)
                        if key not in models:
                            load_start = time.perf_counter()
                            models[key] = clone_encoder(vae, torch.device('cuda', rank))
                            stream.synchronize()
                            stats.setdefault('loads', []).append({'rank': rank, 'model': type(vae.first_stage_model).__name__, 'start': load_start-started, 'seconds': time.perf_counter()-load_start})
                        model = models[key]
                        t = time.perf_counter()
                        x = prepared[i]
                        if getattr(model, 'comfy_has_chunked_io', False):
                            y = model.encode(x, device=torch.device('cuda', rank))
                        else:
                            y = model.encode(x.to(device=rank))
                        results[i] = y.to(device='cpu', dtype=vae.vae_output_dtype())
                        stream.synchronize()
                        stats['items'].append({'index': i, 'rank': rank, 'start': t-started, 'seconds': time.perf_counter()-t, 'shape': list(y.shape)})
                        del y
        finally:
            if stream is not None:
                stream.synchronize()
            models.clear()
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='h3-ref-dp') as pool:
            futures = [pool.submit(worker, rank) for rank in range(2)]
            for future in futures:
                future.result()
    finally:
        cleanup_start = time.perf_counter()
        mm.soft_empty_cache()
        stats['cleanup_seconds'] = time.perf_counter()-cleanup_start
    stats.update(path='dp2', seconds=time.perf_counter()-started)
    logging.info('[H3 Ref DP] %s', json.dumps(stats))
    return results
