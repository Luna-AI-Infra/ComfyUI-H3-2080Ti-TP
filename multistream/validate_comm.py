"""Opt-in real-trajectory communication validation, excluded from timings."""
import collections,json,os,threading
import torch

def install(T):
    from .tp_comm import PeerExchange
    original_init=PeerExchange.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.reference=T.S._Collective('p2p',2,self.devices)
    PeerExchange.__init__=init
    original=T.BlockShard.forward
    calls=collections.Counter();lock=threading.Lock()
    def forward(self,rank,x,emb,segments,rope,options,exchange):
        block=options['block_index'];calls[rank,block]+=1
        if self.communication != 'event_p2p' or block not in (0,25,49):
            return original(self,rank,x,emb,segments,rope,options,exchange)
        communication=self.communication
        self.communication='legacy'
        try: expected=original(self,rank,x.clone(),emb,segments,rope,options,exchange.reference)
        finally:self.communication=communication
        first=original(self,rank,x.clone(),emb,segments,rope,options,exchange)
        second=original(self,rank,x,emb,segments,rope,options,exchange)
        diff=first-expected
        row=dict(step=calls[rank,block],block=block,rank=rank,finite=bool(torch.isfinite(second).all()),
                 max_abs=diff.abs().max().item(),relative_rmse=(diff.square().mean()/expected.square().mean()).sqrt().item(),
                 repeat_max_abs=(first-second).abs().max().item())
        with lock:
            with open(os.environ['H3TP_VALIDATE_COMM'],'a') as f:f.write(json.dumps(row)+'\n')
        if not row['finite'] or row['relative_rmse']>1e-4 or row['repeat_max_abs']!=0:
            raise RuntimeError(f'Communication comparison failed: {row}')
        return second
    T.BlockShard.forward=forward
