"""Anonymous capacity-coupled degree-likelihood readout, not an edge controller.

Exact allocation marginals under a PRODUCT-OF-NODE-DEGREE-LIKELIHOODS working
distribution. This is not the exact graph posterior: incident degrees share
edges. No sampled or persistent node-to-role assignment and no clean node IDs.
"""
import itertools
import math
import time
import numpy as np
import torch
from torch import nn
from scipy.special import logsumexp
from scipy.stats import binom
from role_condition.model import capacities, degrees, features
from late_fusion import LateConditionalDenoiser, LateNodeCondition
from hsgd.bw_preflight.common import schedule

MAX_DP_CELLS=200000

def allocation_marginals(log_likelihood, counts):
    """Exact soft occupancy marginals with integer column capacities.

    All distinguishable node assignments with these multiplicities have equal
    prior mass. Identical roles are already merged by RoleTarget.
    """
    ll=np.asarray(log_likelihood,dtype=np.float64)
    counts=np.asarray(counts)
    if counts.ndim!=1 or not np.issubdtype(counts.dtype,np.integer) or (counts<=0).any():
        raise ValueError("positive integer capacities required")
    n,k=ll.shape
    if k!=len(counts) or counts.sum()!=n or np.isnan(ll).any() or np.isposinf(ll).any():
        raise ValueError("invalid likelihood/capacities")
    if k==1:
        if not np.isfinite(ll[:,0]).all():raise ValueError("impossible role observation")
        return np.ones_like(ll),dict(states=1,cells=n+1,log_partition=float(ll.sum()))
    # Eliminate largest capacity coordinate; this affects cost, not the measure.
    order=np.argsort(counts,kind="stable");inv=np.argsort(order)
    cc=counts[order];ll=ll[:,order]
    size=math.prod(int(x)+1 for x in cc[:-1])
    cells=(n+1)*size
    if cells>MAX_DP_CELLS:raise RuntimeError(f"capacity DP resource limit: {cells}>{MAX_DP_CELLS}; no truncation")
    states=np.array(list(itertools.product(*[range(int(x)+1) for x in cc[:-1]])),dtype=int)
    totals=states.sum(1)
    stride=np.array([math.prod(int(x)+1 for x in cc[j+1:-1]) for j in range(k-1)],dtype=int)
    end=size-1
    forward=np.full((n+1,size),-np.inf);forward[0,0]=0
    transitions=[]
    for i in range(n):
        remaining=i-totals
        valid=(remaining>=0)&(remaining<=cc[-1])
        row=[]
        for j in range(k):
            ok=valid & ((states[:,j]<cc[j]) if j<k-1 else (remaining<cc[-1]))
            src=np.flatnonzero(ok);dst=src+stride[j] if j<k-1 else src
            row.append((src,dst))
            forward[i+1,dst]=np.logaddexp(forward[i+1,dst],forward[i,src]+ll[i,j])
        transitions.append(row)
    z=forward[n,end]
    if not np.isfinite(z):raise ValueError("no feasible assignment under observation support")
    backward=np.full_like(forward,-np.inf);backward[n,end]=0
    result=np.zeros_like(ll)
    for i in range(n-1,-1,-1):
        for j,(src,dst) in enumerate(transitions[i]):
            terms=ll[i,j]+backward[i+1,dst]
            backward[i,src]=np.logaddexp(backward[i,src],terms)
            result[i,j]=np.exp(logsumexp(forward[i,src]+terms)-z)
    if not np.isfinite(result).all():raise FloatingPointError("nonfinite allocation")
    np.testing.assert_allclose(result.sum(1),1,atol=2e-10,rtol=0)
    np.testing.assert_allclose(result.sum(0),cc,atol=2e-9,rtol=0)
    return result[:,inv],dict(states=size,cells=cells,log_partition=float(z))

def degree_logpmf(N,d,a,pi):
    """Exact Bin(d,a+(1-a)pi)+Bin(N-d,(1-a)pi) degree law."""
    if not 0<=d<=N or not 0<=a<=1 or not 0<=pi<=1:raise ValueError("degree support")
    p=a+(1-a)*pi;q=(1-a)*pi
    first=binom.logpmf(np.arange(d+1),d,p)
    second=binom.logpmf(np.arange(N-d+1),N-d,q)
    values=np.full(N+1,-np.inf)
    for observed in range(N+1):
        lo=max(0,observed-(N-d));hi=min(d,observed)
        js=np.arange(lo,hi+1)
        values[observed]=logsumexp(first[js]+second[observed-js])
    if not np.isfinite(logsumexp(values)):raise FloatingPointError("empty degree law")
    np.testing.assert_allclose(logsumexp(values),0,atol=2e-12,rtol=0)
    values.setflags(write=False)
    return values

class CapacityNodeCondition(LateNodeCondition):
    def __init__(self,R,K,width,T):
        super().__init__(R,K,width)
        # The learned query/key attention is REPLACED, not stacked with a loss.
        del self.query;del self.key
        self.T=int(T);self.alpha=schedule(T);self.tables={}

    def forward(self,ctx,t,target):
        started=time.perf_counter();types=ctx.types;K=ctx.model.n_types;R=ctx.model.n_families
        cap=capacities(types,ctx.model.family_pairs);d=degrees(types,ctx.old,ctx.of,R)
        if target is None or target.type_counts!=tuple(map(int,np.bincount(types))) or target.schema!=ctx.model.family_pairs:
            raise ValueError("target/schema mismatch")
        step=int(round(float(t)*self.T))
        if not 0<=step<=self.T or abs(step/self.T-float(t))>1e-8:raise ValueError("time not on frozen schedule")
        xf=features(d,cap[types])
        x=torch.as_tensor(np.c_[xf,np.eye(K,dtype=np.float32)[types],np.full((len(types),1),t,dtype=np.float32)],device=ctx.device)
        statistics_seconds=time.perf_counter()-started;started=time.perf_counter()
        delta=self.degree_out(self.degree_encoder(x));degree_seconds=time.perf_counter()-started
        started=time.perf_counter();context=x.new_zeros((len(types),32));diagnostics=[]
        pi=ctx.model.prior.detach().cpu().numpy()
        for typ,rows in enumerate(target.roles):
            ix=np.flatnonzero(types==typ);vec=np.array([v for v,n in rows]);counts=np.array([n for v,n in rows])
            ll=np.zeros((len(ix),len(rows)))
            for j,v in enumerate(vec):
                for r,N in enumerate(cap[typ]):
                    if not N:continue
                    key=(int(N),int(v[r]),step,float(pi[r]))
                    if key not in self.tables:self.tables[key]=degree_logpmf(int(N),int(v[r]),self.alpha[step],float(pi[r]))
                    ll[:,j]+=self.tables[key][d[ix,r]]
            attention,receipt=allocation_marginals(ll,counts)
            freq=counts/len(ix)
            encoded=self.role_encoder(torch.as_tensor(np.c_[features(vec,cap[typ]),freq,
                np.log1p(counts)/np.log1p(len(ix))].astype(np.float32),device=ctx.device))
            v=self.value(encoded)
            index=torch.as_tensor(ix,device=ctx.device)
            context=context.index_copy(0,index,torch.as_tensor(attention,dtype=x.dtype,device=ctx.device)@v)
            diagnostics.append(dict(type=typ,attention=attention,**receipt))
        if not torch.isfinite(context).all():raise FloatingPointError("nonfinite capacity context")
        return delta,dict(degree=d.copy(),role_context=context,allocation=diagnostics,
            statistics_seconds=statistics_seconds,degree_network_seconds=degree_seconds,
            role_attention_seconds=time.perf_counter()-started)

class CapacityDenoiser(LateConditionalDenoiser):
    architecture_version="anonymous_capacity_degree_likelihood_v1"
    def __init__(self,*args,T=64,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.branch
        self.branch=CapacityNodeCondition(self.n_families,self.n_types,self.backbone.lin_in_X.out_features,T)
        state=old.state_dict()
        self.branch.load_state_dict({k:v for k,v in state.items() if not k.startswith(("query.","key."))},strict=True)
