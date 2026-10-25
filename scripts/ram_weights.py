"""E57 opt-in anonymous pinned weight store; no file-backed tensor mappings."""
import os,json,struct,math,time,hashlib,types,logging
from pathlib import Path
import torch
import ctypes
_LOCKED=[]
STORE={};REPORT=[]
def preload(paths,report):
 for path in paths:
  path=os.path.realpath(path);size=os.path.getsize(path);start=time.monotonic()
  host=torch.empty(size,dtype=torch.uint8,pin_memory=False)
  libc=ctypes.CDLL(None,use_errno=True)
  if libc.mlock(ctypes.c_void_p(host.data_ptr()),ctypes.c_size_t(size))!=0:raise OSError(ctypes.get_errno(),'mlock failed')
  _LOCKED.append(host)
  view=memoryview(host.numpy());digest=hashlib.sha256();offset=0
  with open(path,'rb',buffering=0) as f:
   while offset<size:
    n=f.readinto(view[offset:min(size,offset+64*1024**2)])
    if not n:raise IOError('short weight read: '+path)
    digest.update(view[offset:offset+n]);offset+=n
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])<24*1024**2:raise MemoryError('RAM preload available below24GiB')
  STORE[path]=(host,view)
  REPORT.append(dict(path=path,bytes=size,sha256=digest.hexdigest(),seconds=time.monotonic()-start,pinned=host.is_pinned(),mlocked=True))
  Path(report).write_text(json.dumps(REPORT,indent=2));print('E57_RAM_READY',json.dumps(REPORT[-1]),flush=True)
def install():
 import comfy.utils as u
 old=u.load_torch_file
 def load(ckpt,safe_load=False,device=None,return_metadata=False):
  path=os.path.realpath(ckpt)
  if path not in STORE:
   if str(ckpt).endswith(('.safetensors','.sft','.gguf')):raise RuntimeError('RAM-only weight missing: '+path)
   return old(ckpt,safe_load=safe_load,device=device,return_metadata=return_metadata)
  host,mv=STORE[path];n=struct.unpack('<Q',mv[:8])[0]
  if n>100_000_000 or n+8>len(mv):raise ValueError('invalid safetensors header')
  h=json.loads(bytes(mv[8:8+n]));base=n+8;sd={}
  for k,v in h.items():
   if k=='__metadata__':continue
   start,end=v['data_offsets'];dtype=u._TYPES[v['dtype']]
   if not (0<=start<=end<=len(mv)-base) or end-start!=math.prod(v['shape'])*dtype.itemsize:raise ValueError('invalid tensor bounds: '+k)
   t=torch.frombuffer(mv[base+start:base+end],dtype=dtype).reshape(v['shape']) if end>start else torch.empty(v['shape'],dtype=dtype)
   if device is not None and torch.device(device).type!='cpu':t=t.to(device)
   sd[k]=t
  logging.info('[E57 RAM] safetensors hit %s tensors=%d',path,len(sd))
  # Deliberately do not attach a TensorFileSlice or file-backed storage reference.
  return (sd,h.get('__metadata__',{})) if return_metadata else sd
 u.load_torch_file=load
 import gguf,gguf.gguf_reader as gr
 class NPProxy:
  def __getattr__(self,k):return getattr(gr.np,k)
  def memmap(self,path,mode='r',**kw):
   path=os.path.realpath(path)
   if path not in STORE:raise RuntimeError('RAM-only GGUF missing: '+path)
   if mode not in ('r','c'):raise ValueError('RAM GGUF only supports read/copy modes')
   logging.info('[E57 RAM] GGUF hit %s',path)
   return STORE[path][0].numpy()
 original=gguf.GGUFReader.__init__;g=dict(original.__globals__);g['np']=NPProxy()
 gguf.GGUFReader.__init__=types.FunctionType(original.__code__,g,original.__name__,original.__defaults__,original.__closure__)
 print('E57_RAM_LOADER_INSTALLED',flush=True)
