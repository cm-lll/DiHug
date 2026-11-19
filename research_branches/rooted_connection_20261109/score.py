"""One CPU hard-graph output candidate: anchored typed connection profiles.

No RolX, learned representation, input change, or probability relaxation.
Root features concatenate relation degree, typed triangle participation and
non-backtracking relation-pair walk counts. Every block shares the SAME root.
"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/hard_structure_rebar_cpu_20261108'))
sys.path.insert(0,str(ROOT/'research_branches/hard_structure_incremental_cpu_20261108'))
from objective import JointLocalScore
from incremental import IncrementalScore,kernel

class RootedScore(JointLocalScore):
    version='rooted_connection_joint_degree_triangle_nonbacktracking_v1'
    def __init__(self,types,query,families,reference,n_relations):
        nt=len(np.unique(types));self.base_width=n_relations+nt*(nt+1)//2
        super().__init__(types,query,families,reference,n_relations)
        # Fixed equal-block distance, reference-only RMS scaling; no fit/search.
        with torch.no_grad():
            raw=self.features(self.reference)
            for t,ix in self.nodes.items():
                mask=self.masks[t];cols=torch.where(mask)[0];d=len(cols)
                groups=[cols<self.r,(cols>=self.r)&(cols<self.base_width),cols>=self.base_width]
                active=[g for g in groups if bool(g.any())]
                for group in active:
                    self.scales[t][group]*=(len(active)*int(group.sum())/max(1,d))**.5
                self.targets[t]=torch.log1p(raw[ix][:,mask])/self.scales[t]
                self.target_self[t]=self.kernel(self.targets[t],self.targets[t]).mean()

    def features(self,weights):
        base=super().features(weights)
        degree=base[...,:self.r]
        flat=weights.new_zeros(*weights.shape[:-1],self.r*self.n*self.n)
        i,j=self.query;f=self.families
        flat=flat.index_add(-1,f*self.n*self.n+i*self.n+j,weights)
        flat=flat.index_add(-1,f*self.n*self.n+j*self.n+i,weights)
        adjacency=flat.reshape(*weights.shape[:-1],self.r,self.n,self.n)
        neighbors=adjacency@degree.unsqueeze(-3)
        neighbors=neighbors.transpose(-3,-2)
        eye=torch.eye(self.r,dtype=weights.dtype,device=weights.device)
        neighbors=neighbors-degree.unsqueeze(-1)*eye
        # Meaningful counts only on HARD graphs. No claim for fractional weights.
        return torch.cat([base,neighbors.flatten(-2)],dim=-1)

class RootedIncremental(IncrementalScore):
    def __init__(self,core,bits):
        super().__init__(core,bits)
        self.start=core.base_width
        self.edge_family=np.full((self.n,self.n),-1,dtype=int)
        live=np.flatnonzero(self.bits)
        u,v=self.query[:,live]
        self.edge_family[u,v]=self.edge_family[v,u]=self.families[live]

    def raw_flips(self,indices):
        ix=np.asarray(indices,dtype=int);i,j=self.query[:,ix];rel=self.families[ix]
        sign=1-2*self.bits[ix];b=np.arange(len(ix));common=self.adjacency[i]&self.adjacency[j]
        changed=self.adjacency[i]|self.adjacency[j]
        changed[b,i]=True;changed[b,j]=True
        raw=np.broadcast_to(self.raw,(len(ix),*self.raw.shape)).copy()
        raw[b,i,rel]+=sign;raw[b,j,rel]+=sign
        bb,kk=np.nonzero(common)
        if len(bb):
            for endpoint,ta,tb in [(i[bb],self.types[j[bb]],self.types[kk]),(j[bb],self.types[i[bb]],self.types[kk]),(kk,self.types[i[bb]],self.types[j[bb]])]:
                np.add.at(raw,(bb,endpoint,self.triangle_channel[ta,tb]),sign[bb])
        # N^{r,s}=A_r d_s - 1[r=s] d_r. All finite cross terms retained.
        for endpoint in [i,j]:
            bb,kk=np.nonzero(self.adjacency[endpoint])
            rr=self.edge_family[endpoint[bb],kk]
            np.add.at(raw,(bb,kk,self.start+rr*self.r+rel[bb]),sign[bb])
        cols=self.start+rel[:,None]*self.r+np.arange(self.r)[None,:]
        raw[b[:,None],i[:,None],cols]+=sign[:,None]*self.raw[j,:self.r]
        raw[b[:,None],j[:,None],cols]+=sign[:,None]*self.raw[i,:self.r]
        diag=self.start+rel*self.r+rel
        raw[b,i,diag]+=1-sign;raw[b,j,diag]+=1-sign
        if raw.min()<0:raise ArithmeticError('negative non-backtracking count')
        return raw,changed

    def differences(self,indices=None,batch_size=64):
        indices=np.arange(len(self.bits)) if indices is None else np.asarray(indices,dtype=int)
        result=np.empty(len(indices))
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size];raw_new,changed=self.raw_flips(ix)
            delta=np.zeros(len(ix))
            for cache in self.cache.values():
                nodes,mask=cache['nodes'],cache['mask']
                if not mask.any():continue
                local=changed[:,nodes];counts=local.sum(-1);width=int(counts.max())
                if width==0:continue
                row,col=np.nonzero(local);offsets=np.r_[0,np.cumsum(counts)[:-1]]
                position=np.arange(len(row))-offsets[row]
                gather=np.zeros((len(ix),width),dtype=int);gather[row,position]=col
                valid=np.arange(width)[None,:]<counts[:,None]
                updated=raw_new[np.arange(len(ix))[:,None],nodes[gather]][...,mask]
                xnew=np.log1p(updated)/cache['scale'];xold=cache['x'][gather]
                row_delta=((kernel(xnew,cache['x']).sum(-1)-cache['xx_row'][gather])*valid).sum(-1)
                valid_pair=valid[:,:,None]&valid[:,None,:]
                correction=((kernel(xnew,xnew)-2*kernel(xnew,xold)+cache['xx'][gather[:,:,None],gather[:,None,:]])*valid_pair).sum((-1,-2))
                cross=((kernel(xnew,cache['y']).sum(-1)-cache['xy_row'][gather])*valid).sum(-1)
                delta+=(2*row_delta+correction-2*cross)/len(nodes)**2
            result[start:start+len(ix)]=(1-2*self.bits[ix])*delta/len(self.cache)
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite conditional edge difference')
        return result
