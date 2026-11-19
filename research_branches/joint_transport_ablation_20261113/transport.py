"""Frozen type-preserving joint degree/heat quadratic assignment cost.

No reference node-ID alignment, count penalty, graph repair, learnt parameters,
model calls, bandwidth search, or dependence on evaluation/generated graphs.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('transport_heat',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py')
heatmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(heatmod)

def bandwidth(x):
    ds=pdist(x,'sqeuclidean');positive=ds[ds>1e-24]
    return float(np.median(positive)) if len(positive) else 1.

def assignment(x,y):
    cost=cdist(x,y,'sqeuclidean');ii,jj=linear_sum_assignment(cost)
    return float(cost[ii,jj].mean()),jj

class TransportScore:
    def __init__(self,ds,case,query,fam,labels):
        self.ds=ds;self.types=np.asarray(case['space'].types);self.n=len(self.types);self.r=case['n_families'];self.q=np.asarray(query);self.f=np.asarray(fam)
        self.labels=np.asarray(labels,dtype=np.uint8);self.target=np.bincount(self.f,weights=self.labels,minlength=self.r)
        self.prior=np.asarray(case['prior']);self.rh,self.rd=self.features(self.labels)
        self.scales={};self.reference={}
        for typ in np.unique(self.types):
            sel=self.types==typ;scale=np.maximum(1.,self.rd[sel].mean(0));raw=self.rd[sel]/scale
            db=bandwidth(raw);hb=bandwidth(self.rh[sel]);self.scales[int(typ)]=(scale,db,hb)
            self.reference[int(typ)]=np.concatenate([raw/np.sqrt(db),self.rh[sel]/np.sqrt(hb)],axis=1)
        self.s=dict(version='raw_joint_degree_heat_W2_squared_v1',degree='raw relation-degree / max(1,reference per-type mean)',
            block_scales='separate reference within-type median positive squared distances; fallback 1',
            comparison='same type optimal bijection, joint concatenation, squared distance, equal type mean',
            standalone_count_penalty=False,coefficients=dict(degree=1.,heat=1.),
            by_type={str(t):dict(degree_scale=x[0].tolist(),degree_bandwidth_squared=x[1],heat_bandwidth_squared=x[2]) for t,x in self.scales.items()})
        self.last=None
    def features(self,b):
        b=np.asarray(b);assert b.shape==(self.q.shape[1],) and np.isin(b,[0,1]).all()
        for r,p in enumerate(self.prior):
            if p==0:assert not np.any(b[self.f==r]),'fixed-zero support violated'
            if p==1:assert np.all(b[self.f==r]),'fixed-one support violated'
        sel=b.astype(bool);a=np.zeros((self.n,self.n));a[self.q[0,sel],self.q[1,sel]]=1;a+=a.T
        assert not np.diag(a).any() and int(a.sum())==2*int(b.sum())
        d=np.zeros((self.n,self.r));np.add.at(d,(self.q[0,sel],self.f[sel]),1);np.add.at(d,(self.q[1,sel],self.f[sel]),1)
        return heatmod.heat(a,self.types)['feat'],d
    def __call__(self,b):
        h,d=self.features(b);terms=[];parts=[]
        for typ in np.unique(self.types):
            sel=self.types==typ;scale,db,hb=self.scales[int(typ)];x=np.concatenate([d[sel]/scale/np.sqrt(db),h[sel]/np.sqrt(hb)],axis=1);y=self.reference[int(typ)]
            total,order=assignment(x,y);delta=x-y[order];dc=float(np.mean(np.sum(delta[:,:self.r]**2,axis=1)));hc=float(np.mean(np.sum(delta[:,self.r:]**2,axis=1)))
            lower=float(np.sum(((d[sel].mean(0)-self.rd[sel].mean(0))/scale)**2)/db)
            assert total+1e-10>=lower and abs(total-dc-hc)<1e-9
            terms.append(total);parts.append(dict(type=int(typ),joint_cost=total,degree_at_joint_match=dc,heat_at_joint_match=hc,count_mean_lower_bound=lower))
        self.last=parts
        # Worker historical interface has two scalar components. The second is
        # exactly zero; it is NOT a hidden edge-count or extra structural term.
        return float(np.mean(terms)),0.,parts
