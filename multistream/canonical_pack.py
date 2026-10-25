"""Immutable, per-rank CPU master; regrouping changes views, never weight bytes."""
import torch
from .tp_residency import skeleton, tensor_fields, assign, PackedGroup

class CanonicalPack:
    def __init__(self, count, make_block, pin_memory=True):
        self.templates=[];self.layout=[];self.boundaries=[0]
        offset=0
        # Two bounded passes avoid retaining a complete extra rank of staged weights.
        for i in range(count):
            block=make_block(i);template=skeleton(block)
            for owner,field,t in tensor_fields(block):
                if t.device.type!='cpu':raise ValueError('Canonical CPU master requires CPU tensors')
                offset=(offset+255)//256*256
                size=t.numel()*t.element_size()
                self.layout.append((i,owner,field,offset,size,t.dtype,tuple(t.shape)))
                offset+=size;assign(template,owner,field,None)
            offset=(offset+255)//256*256
            self.boundaries.append(offset);self.templates.append(template)
            del block
        self.host=torch.empty(offset,dtype=torch.uint8,pin_memory=pin_memory)
        # Initialize padding as well, so whole-buffer hashes are deterministic.
        self.host.zero_()
        self.by_block=[[] for _ in range(count)]
        for item in self.layout:self.by_block[item[0]].append(item)
        for i in range(count):
            block=make_block(i);fields=list(tensor_fields(block));spec=self.by_block[i]
            if len(fields)!=len(spec):raise ValueError('Block layout changed while packing')
            for (owner,field,t),(_,o,f,start,size,dtype,shape) in zip(fields,spec):
                if (owner,field,t.dtype,tuple(t.shape))!=(o,f,dtype,shape):raise ValueError('Block layout changed while packing')
                self.host[start:start+size].view(dtype).view(shape).copy_(t)
            del fields,block

    def groups(self, ranges):
        cursor=0;result=[]
        for start,stop in ranges:
            if start!=cursor or not start<stop<=len(self.templates):raise ValueError('Groups must partition all blocks contiguously')
            begin,end=self.boundaries[start],self.boundaries[stop]
            group=PackedGroup.__new__(PackedGroup)
            group.host=self.host[begin:end]  # owns a view, no allocation/copy
            group.templates=[skeleton(t) for t in self.templates[start:stop]]
            group.layout=[(i-start,o,f,offset-begin,size,dtype,shape)
                          for i in range(start,stop) for _,o,f,offset,size,dtype,shape in self.by_block[i]]
            result.append(group);cursor=stop
        if cursor!=len(self.templates):raise ValueError('Incomplete block partition')
        return result
