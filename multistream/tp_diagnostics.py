"""E62 opt-in per-stage CUDA error localization; never enabled by default."""
import contextlib,os,threading
import torch
from .log import log

def install_stage(block,rank,index):
    if os.environ.get('H3_TP_DIAGNOSTIC_SYNC','0')!='1':return
    @contextlib.contextmanager
    def stage(name):
        try:
            yield
            torch.cuda.synchronize(rank)
        except BaseException:
            log.exception('[H3 DIAG] rank=%d block=%d stage=%s',rank,index,name)
            raise
    block.stage=stage
