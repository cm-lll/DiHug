"""Exact hard-graph local metric changes, independent of the MMD scorer."""
from collections import Counter
import numpy as np

class Metrics:
 def __init__(self,types,q,f,y,b,r):
  self.types=np.asarray(types);self.q=np.asarray(q);self.f=np.asarray(f);self.b=np.asarray(b,dtype=int);self.r=r;self.n=len(types);self.ts=sorted(set(types.tolist()));self.nt=len(self.ts)
  self.a=np.zeros((self.n,self.n),dtype=bool);i,j=q;self.a[i,j]=self.b;self.a[j,i]=self.b
  self.target=np.zeros_like(self.a);self.target[i,j]=y;self.target[j,i]=y
  self.degree=self.a.sum(1).astype(int);self.ref_degree=self.target.sum(1).astype(int)
  self.triangles=int(np.trace(self.a.astype(np.int64)@self.a.astype(np.int64)@self.a.astype(np.int64))//6)
  self.ref_triangles=int(np.trace(self.target.astype(np.int64)@self.target.astype(np.int64)@self.target.astype(np.int64))//6)
  self.vec=np.zeros((self.n,r),dtype=int);self.refvec=np.zeros_like(self.vec)
  for ends in q:np.add.at(self.vec,(ends,f),self.b);np.add.at(self.refvec,(ends,f),y)
  self.cdf={};self.diff={};self.size={};self.counters={}
  for t in self.ts:
   ix=types==t;self.size[t]=int(ix.sum());h=np.bincount(self.degree[ix],minlength=self.n);hr=np.bincount(self.ref_degree[ix],minlength=self.n);self.cdf[t]=np.cumsum(h-hr,dtype=float)/self.size[t]
   c=Counter(map(tuple,self.vec[ix]));cr=Counter(map(tuple,self.refvec[ix]));self.counters[t]=c
   self.diff[t]={k:c.get(k,0)-cr.get(k,0) for k in c.keys()|cr.keys()}
  self.base_w1=float(np.mean([abs(x).sum() for x in self.cdf.values()]));self.base_tv=float(np.mean([.5*sum(abs(v) for v in self.diff[t].values())/self.size[t] for t in self.ts]))
 def changes(self):
  size=self.q.shape[1];dw=np.empty(size);dtv=np.empty(size);cn=np.sum(self.a[self.q[0]]&self.a[self.q[1]],axis=1);sign=1-2*self.b
  dtri=abs(self.triangles+sign*cn-self.ref_triangles)-abs(self.triangles-self.ref_triangles)
  for e,(i,j) in enumerate(self.q.T):
   s=int(sign[e]);f=int(self.f[e]);bins=Counter();vectors=Counter()
   for u in [i,j]:
    t=int(self.types[u]);k=int(self.degree[u] if s==1 else self.degree[u]-1);bins[(t,k)]-=s
    old=tuple(self.vec[u]);v=self.vec[u].copy();v[f]+=s;assert v.min()>=0;new=tuple(v);vectors[(t,old)]-=1;vectors[(t,new)]+=1
   dw[e]=sum(abs(self.cdf[t][k]+v/self.size[t])-abs(self.cdf[t][k]) for (t,k),v in bins.items())/self.nt
   dtv[e]=.5*sum((abs(self.diff[t].get(k,0)+v)-abs(self.diff[t].get(k,0)))/self.size[t] for (t,k),v in vectors.items())/self.nt
  return dict(triangle_abs_error=dtri.astype(float),total_degree_W1=dw,joint_degree_TV=dtv,common_neighbors=cn,flip_sign=sign)

def brute(types,q,f,y,b,r):
 n=len(types);a=np.zeros((n,n),dtype=np.int64);ref=np.zeros_like(a);a[q[0],q[1]]=b;a[q[1],q[0]]=b;ref[q[0],q[1]]=y;ref[q[1],q[0]]=y
 # Independent triangle enumeration, sorted degree W1 and empirical tuple distributions.
 tri=sum(a[i,j]*a[j,k]*a[i,k] for i in range(n) for j in range(i+1,n) for k in range(j+1,n));rt=sum(ref[i,j]*ref[j,k]*ref[i,k] for i in range(n) for j in range(i+1,n) for k in range(j+1,n))
 v=np.zeros((n,r),dtype=int);rv=np.zeros_like(v)
 for e,(i,j) in enumerate(q.T):v[i,f[e]]+=int(b[e]);v[j,f[e]]+=int(b[e]);rv[i,f[e]]+=int(y[e]);rv[j,f[e]]+=int(y[e])
 w=[];tv=[]
 for t in sorted(set(types.tolist())):
  ix=types==t;w.append(np.mean(abs(np.sort(a.sum(1)[ix])-np.sort(ref.sum(1)[ix]))));x=Counter(map(tuple,v[ix]));z=Counter(map(tuple,rv[ix]));tv.append(.5*sum(abs(x.get(k,0)-z.get(k,0)) for k in x.keys()|z.keys())/int(ix.sum()))
 return dict(triangle_abs_error=float(abs(tri-rt)),total_degree_W1=float(np.mean(w)),joint_degree_TV=float(np.mean(tv)))
