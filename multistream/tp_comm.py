# SPDX-License-Identifier: GPL-3.0-only
"""Native-allocation P2P exchanges scoped to one resident TP forward."""
import threading
import torch
import comfy.memory_management
from . import split as S


class PeerExchange:
    def __init__(self, devices):
        if comfy.memory_management.aimdo_enabled or torch.cuda.memory.get_allocator_backend() != 'native':
            raise ValueError('Event P2P requires --disable-dynamic-vram --disable-cuda-malloc')
        from cuda.bindings import runtime
        self.runtime = runtime
        self.devices = devices
        S._enable_peer_access(devices)
        self.streams = [torch.cuda.Stream(device=d) for d in devices]
        self.barrier = threading.Barrier(2, timeout=120)
        self.slots = [{}, {}]
        self.sequence = [0, 0]

    def _copy(self, source, rank, ready):
        device, stream = self.devices[rank], self.streams[rank]
        dest = torch.empty_like(source, device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        stream.wait_event(ready)
        with torch.cuda.device(device):
            status, = self.runtime.cudaMemcpyPeerAsync(
                dest.data_ptr(), device.index, source.data_ptr(), source.device.index,
                source.numel() * source.element_size(), stream.cuda_stream)
            if status != self.runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f'cudaMemcpyPeerAsync failed: {status}')
            source.record_stream(stream)
            dest.record_stream(stream)
            done = torch.cuda.Event()
            done.record(stream)
        return dest, done

    def copy_to(self, tensor, rank):
        if not isinstance(tensor, torch.Tensor) or tensor.device == self.devices[rank]:
            return tensor
        if tensor.device.type != 'cuda':
            return tensor.to(self.devices[rank])
        tensor = tensor.contiguous()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(tensor.device))
        result, done = self._copy(tensor, rank, ready)
        torch.cuda.current_stream(self.devices[rank]).wait_event(done)
        return result

    def start(self, rank, value):
        value = value.contiguous()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.devices[rank]))
        index = self.sequence[rank]
        self.sequence[rank] += 1
        self.slots[rank][index] = (value, ready)
        self.barrier.wait()
        peer, peer_ready = self.slots[1-rank].pop(index)
        received, done = self._copy(peer, rank, peer_ready)
        return rank, value, received, done

    def finish(self, handle, swiglu=False):
        rank, own, peer, done = handle
        torch.cuda.current_stream(self.devices[rank]).wait_event(done)
        parts = [own, peer] if rank == 0 else [peer, own]
        if swiglu:
            a, b = [p.chunk(2, -1) for p in parts]
            parts = [a[0], b[0], a[1], b[1]]
        return torch.cat(parts, -1)

    def gather(self, rank, value):
        return self.finish(self.start(rank, value))

    def abort(self):
        self.barrier.abort()

    def close(self):
        for stream in self.streams:
            stream.synchronize()
