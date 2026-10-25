"""Opt-in real-trajectory comparisons. Never use these runs as timing results."""
import collections
import json
import os
import threading

import torch


def install(T):
    original = T.BlockShard.forward
    calls = collections.Counter()
    lock = threading.Lock()

    def forward(self, rank, x, emb, segments, rope, options, exchange):
        block = options['block_index']
        calls[rank, block] += 1
        step = calls[rank, block]
        if block not in (0, 25, 49) or step not in (1, 10, 20):
            return original(self, rank, x, emb, segments, rope, options, exchange)
        backend = self.attention_backend
        self.attention_backend = 'pytorch_fp32'
        try:
            expected = original(self, rank, x.clone(), emb, segments, rope, options, exchange)
        finally:
            self.attention_backend = backend
        output = original(self, rank, x, emb, segments, rope, options, exchange)
        diff = output - expected
        row = dict(step=step, block=block, rank=rank, backend=backend,
                   finite=bool(torch.isfinite(output).all()),
                   max_abs=diff.abs().max().item(),
                   relative_rmse=(diff.square().mean()/expected.square().mean()).sqrt().item())
        with lock:
            with open(os.environ['H3TP_VALIDATE'], 'a') as f:
                f.write(json.dumps(row)+'\n')
        if not row['finite'] or row['relative_rmse'] > 0.005:
            raise RuntimeError(f'H3 TP real-trajectory precision check failed: {row}')
        return output

    T.BlockShard.forward = forward
