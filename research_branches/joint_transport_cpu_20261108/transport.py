"""Type-preserving empirical joint W1, exact assignment and exact edge flips.
Raw counts/reference-only scaling; no learned parameters, entropic solver or GPU.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def assignment_value(cost):
 rows,cols=linear_sum_assignment(cost)
 assert len(rows)==len(cost) and len(np.unique(cols))==len(cost)
 return float(cost[rows,cols].mean())

class JointTransport:
 def __init__(self,core,bits):
  import torch
  self.core=core;self.types=core.types;self.q=core.query.numpy();self.f=core.families.numpy();self.bits=np.asarray(bits,dtype=float);self.n=core.n;self.r=core.r;self.nt=len(core.nodes)
  assert np.isin(self.bits,[0,1]).all()
  self.a=np.zeros((self.n,self.n),dtype=bool);i,j=self.q;self.a[i,j]=self.bits;self.a[j,i]=self.bits
  with torch.no_grad():self.raw=core.features(torch.from_numpy(self.bits)).numpy();self.refraw=core.features(core.reference).numpy()
  self.channel={tuple(pair):self.r+k for k,pair in enumerate(core.triangle_pairs)};self.cache={};self.parts={}
  for t,indices in core.nodes.items():
   ix=indices.numpy();mask=core.masks[t].numpy();weights=np.zeros(self.raw.shape[1]);scale=np.maximum(1.,self.refraw[ix].mean(0));blocks=[np.flatnonzero(mask & (np.arange(len(mask))<self.r)),np.flatnonzero(mask & (np.arange(len(mask))>=self.r))];blocks=[b for b in blocks if len(b)]
   for b in blocks:weights[b]=1/(len(blocks)*len(b)*scale[b])
   target=self.refraw[ix]*weights;cost=cdist(self.raw[ix]*weights,target,metric='cityblock');value=assignment_value(cost);self.parts[t]=value
   self.cache[t]=dict(nodes=ix,weights=weights,scale=scale,target=target,cost=cost,block_count=len(blocks),active_degree=int(sum(mask[:self.r])),active_triangle=int(sum(mask[self.r:])))
  self.base=float(np.mean(list(self.parts.values())))
 def value(self,raw):
  parts={t:assignment_value(cdist(raw[c['nodes']]*c['weights'],c['target'],metric='cityblock')) for t,c in self.cache.items()}
  return float(np.mean(list(parts.values()))),parts
 def bounds(self,parts):
  degree=0.;tri=0.
  for t,c in self.cache.items():
   w=c['weights'];wd=w[:self.r];wt=w[self.r:]
   if np.any(wd>0):degree+=parts[t]/wd[wd>0].min()/self.nt
   if np.any(wt>0):tri+=len(c['nodes'])*parts[t]/wt[wt>0].min()/3
  return dict(total_degree_W1=degree,joint_degree_TV=degree,triangle_abs_error=tri)
 def flipped_raw(self,e):
  i,j=self.q[:,e];s=1-2*self.bits[e];f=self.f[e];common=np.flatnonzero(self.a[i]&self.a[j]);raw=self.raw.copy();raw[i,f]+=s;raw[j,f]+=s
  for k in common:
   for center,u,v in [(i,j,k),(j,i,k),(k,i,j)]:raw[center,self.channel[tuple(sorted((self.types[u],self.types[v])))]]+=s
  assert raw.min()>=0
  return raw,np.unique(np.r_[i,j,common])
 def differences(self,indices,return_bounds=False):
  flips=[];bounds=[]
  for e in indices:
   raw,changed=self.flipped_raw(int(e));parts=self.parts.copy()
   for t,c in self.cache.items():
    local=np.flatnonzero(np.isin(c['nodes'],changed))
    if not len(local):continue
    cost=c['cost'].copy();cost[local]=cdist(raw[c['nodes'][local]]*c['weights'],c['target'],metric='cityblock');parts[t]=assignment_value(cost)
   flips.append(np.mean(list(parts.values()))-self.base)
   if return_bounds:bounds.append(self.bounds(parts))
  result=np.asarray(flips);assert np.isfinite(result).all()
  return (result,bounds) if return_bounds else result
