"""E54 bounded empirical workspace policy, not a universal OOM guarantee."""
import math
GIB = 2**30

def ranges(count, groups):
    if not 3 <= groups <= count: raise ValueError('num_groups must be in [3, block_count]')
    q, r = divmod(count, groups)
    sizes = [q + (i < r) for i in range(groups)]
    starts = [sum(sizes[:i]) for i in range(groups)]
    return [(a, a+b) for a,b in zip(starts,sizes)]

def choose(count, block_bytes, tokens, free_bytes, reserve_gib=4, safety_gib=2, max_slots=8):
    if not 2 <= max_slots <= 49: raise ValueError('max_slots must be in [2, 49]')
    if not 0 < tokens <= 48000:
        raise ValueError('auto_tokens calibrated domain is 1..48000 packed tokens; use manual after validation')
    # E53 empirical envelope: constant workspace + linear token buffers.
    # Kept ABOVE measured incremental live bytes; safety additionally covers
    # caching/fragmentation, contexts and concurrent allocation uncertainty.
    working = max(reserve_gib, 3.0 + 0.00020 * tokens)
    budget = min(free_bytes) - (working + safety_gib) * GIB
    candidates=[]
    for groups in range(3, min(count,25)+1):
        rr=ranges(count,groups)
        capacity=max(b-a for a,b in rr)*max(block_bytes)
        for slots in range(2,min(groups,max_slots+1)):
            weight=capacity*slots
            # Approximate SM75 Kitchen50: conservative compute window and H2D.
            compute_s=(slots-1)*min(b-a for a,b in rr)*0.20*(tokens/17000)
            copy_s=capacity/(5*GIB)
            candidates.append(dict(groups=groups,slots=slots,weight_bytes=weight,
                fits=weight<=budget,compute_window_s=compute_s,copy_s=copy_s,
                predicted_wait_s=max(0,copy_s-compute_s)*groups))
    fit=[c for c in candidates if c['fits']]
    if not fit: raise RuntimeError('No residency plan fits estimated workspace + safety reserve')
    # Hide copies first, then minimize transfer launch count, then memory.
    best=min(fit,key=lambda c:(c['predicted_wait_s'],c['groups'],c['weight_bytes']))
    return dict(best, tokens=tokens,workspace_gib=working,safety_gib=safety_gib,
                budget_bytes=budget,candidates=candidates,max_slots=max_slots,policy='e54-v1-empirical')
