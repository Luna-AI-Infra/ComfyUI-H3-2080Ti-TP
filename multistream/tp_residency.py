# SPDX-License-Identifier: GPL-3.0-only
"""Packed pinned CPU masters and two GPU slots for three TP block groups."""
import copy
import torch


def tensor_fields(block):
    for name in ('qkv', 'out', 'fc1', 'fc2', 'adaln'):
        linear = getattr(block, name)
        for field in ('weight', 'scale', 'bias', 'lora_down', 'lora_up'):
            value = getattr(linear, field)
            if value is not None:
                yield name, field, value
    for name in ('n1', 'n2', 'qn', 'kn'):
        yield None, name, getattr(block, name)


def skeleton(block):
    result = copy.copy(block)
    for name in ('qkv', 'out', 'fc1', 'fc2', 'adaln'):
        setattr(result, name, copy.copy(getattr(block, name)))
    return result


def assign(block, owner, field, value):
    setattr(getattr(block, owner) if owner else block, field, value)


class PackedGroup:
    def __init__(self, blocks):
        self.templates = [skeleton(b) for b in blocks]
        self.layout = []
        offset = 0
        for i, block in enumerate(blocks):
            for owner, field, tensor in tensor_fields(block):
                assert tensor.device.type == 'cpu'
                offset = (offset + 255) // 256 * 256
                size = tensor.numel() * tensor.element_size()
                self.layout.append((i, owner, field, offset, size, tensor.dtype, tuple(tensor.shape)))
                offset += size
        self.host = torch.empty(offset, dtype=torch.uint8, pin_memory=True)
        for i, owner, field, start, size, dtype, shape in self.layout:
            tensor = getattr(getattr(blocks[i], owner) if owner else blocks[i], field)
            self.host[start:start + size].view(dtype).view(shape).copy_(tensor)
            assign(self.templates[i], owner, field, None)

    def bind(self, buffer):
        blocks = [skeleton(b) for b in self.templates]
        for i, owner, field, start, size, dtype, shape in self.layout:
            assign(blocks[i], owner, field, buffer[start:start + size].view(dtype).view(shape))
        return blocks


class TwoSlotBank:
    """Configurable FIFO ring. CPU masters immutable; CUDA events own reuse."""
    def __init__(self, groups, device, slots=2, preload='eager'):
        if not 2 <= slots < len(groups):
            raise ValueError('Need 2 <= resident_slots < num_groups')
        self.groups, self.device = groups, device
        self.slots = slots
        self.final_step = False
        self.pending = None
        capacity = max(g.host.numel() for g in groups)
        with torch.cuda.device(device):
            self.buffers = [torch.empty(capacity, dtype=torch.uint8, device=device) for _ in range(slots)]
            self.stream = torch.cuda.Stream(device=device)
            self.ready = [torch.cuda.Event() for _ in range(slots)]
            self.done = [None] * slots
            self.loaded = [None] * slots
            self.views = [None] * slots
            for group in range(slots):
                self.prefetch(group, group)

    def prefetch(self, group, slot):
        if self.loaded[slot] == group:
            return
        with torch.cuda.stream(self.stream):
            if self.done[slot] is not None:
                self.stream.wait_event(self.done[slot])
            source = self.groups[group]
            self.buffers[slot][:source.host.numel()].copy_(source.host, non_blocking=True)
            self.ready[slot].record(self.stream)
        self.views[slot] = source.bind(self.buffers[slot])
        self.loaded[slot] = group

    def acquire(self, group, prefetch_next=True):
        # ResidentTP synchronizes the device at each step boundary. Do not
        # leave a needless future-group H2D there: issue it alongside the
        # next step's first group instead, preserving S-slot lookahead.
        if self.pending is not None:
            target, slot = self.pending
            self.pending = None
            self.prefetch(target, slot)
        if group not in self.loaded:
            raise RuntimeError(f'TP prefetch lost group {group}: {self.loaded}')
        slot = self.loaded.index(group)
        torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        self.current_group = group
        return slot, self.views[slot]

    def finish(self, slot):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.done[slot] = event
        target = self.current_group + self.slots
        if not self.final_step or target < len(self.groups):
            if self.current_group == len(self.groups) - 1:
                self.pending = (target % len(self.groups), slot)
            else:
                self.prefetch(target % len(self.groups), slot)

    def close(self):
        self.stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.views = []; self.buffers = []; self.groups = []
