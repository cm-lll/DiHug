"""Float64 CUDA evaluator of the SAME finite hard-graph score differences."""
import numpy as np
import torch
from score import RootedIncremental

def kernel(x,y):
    d=max(x.shape[-1],1)
    distance=x.square().sum(-1).unsqueeze(-1)+y.square().sum(-1).unsqueeze(-2)-2*(x@y.transpose(-1,-2))
    return torch.exp(-distance.clamp_min(0)/(2*d))

class GPUIncremental(RootedIncremental):
    def __init__(self,core,bits,device='cuda'):
        super().__init__(core,bits);self.device=torch.device(device)
        self.gcache={}
        for t,c in self.cache.items():
            self.gcache[t]={k:torch.as_tensor(v,device=self.device) for k,v in c.items() if k not in ['nodes','mask']}

    @torch.no_grad()
    def differences(self,indices=None,batch_size=128):
        indices=np.arange(len(self.bits)) if indices is None else np.asarray(indices,dtype=int)
        result=torch.empty(len(indices),dtype=torch.float64,device=self.device)
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size];raw_new,changed=self.raw_flips(ix)
            delta=torch.zeros(len(ix),dtype=torch.float64,device=self.device)
            for t,cache in self.cache.items():
                nodes,mask=cache['nodes'],cache['mask']
                if not mask.any():continue
                local=changed[:,nodes];counts=local.sum(-1);width=int(counts.max())
                if width==0:continue
                row,col=np.nonzero(local);offsets=np.r_[0,np.cumsum(counts)[:-1]]
                position=np.arange(len(row))-offsets[row]
                gather=np.zeros((len(ix),width),dtype=int);gather[row,position]=col
                valid=np.arange(width)[None,:]<counts[:,None]
                updated=raw_new[np.arange(len(ix))[:,None],nodes[gather]][...,mask]
                c=self.gcache[t];g=torch.as_tensor(gather,device=self.device);v=torch.as_tensor(valid,device=self.device)
                xnew=torch.log1p(torch.as_tensor(updated,device=self.device))/c['scale'];xold=c['x'][g]
                row_delta=((kernel(xnew,c['x']).sum(-1)-c['xx_row'][g])*v).sum(-1)
                correction=((kernel(xnew,xnew)-2*kernel(xnew,xold)+c['xx'][g[:,:,None],g[:,None,:]])*(v[:,:,None]&v[:,None,:])).sum((-1,-2))
                cross=((kernel(xnew,c['y']).sum(-1)-c['xy_row'][g])*v).sum(-1)
                delta+=(2*row_delta+correction-2*cross)/len(nodes)**2
            result[start:start+len(ix)]=torch.as_tensor(1-2*self.bits[ix],device=self.device)*delta/len(self.cache)
        out=result.cpu().numpy()
        if not np.isfinite(out).all():raise FloatingPointError('nonfinite CUDA score difference')
        return out
