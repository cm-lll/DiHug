"""Experimental CPU ground cost. Not imported by production training.

Reference-scaled degree cost stays quadratic. The entire node heat residual
uses twice pseudo-Huber(delta=1), BEFORE joint Hungarian matching.
This is not truncated/unbalanced/outlier-removal OT, nor a W2 metric claim.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def robust_heat_squared(q):
    q=np.asarray(q,dtype=np.float64)
    if not np.isfinite(q).all() or np.any(q<0):raise ValueError('invalid squared heat residual')
    return 2*q/(np.sqrt(1+q)+1)

def evaluate(sc,b):
    h,d=sc.features(b);allparts={False:[],True:[]};nodes=[]
    for typ,(scale,db,hb) in sc.scales.items():
        ix=np.flatnonzero(sc.types==typ)
        gd=cdist(d[ix]/scale/np.sqrt(db),sc.rd[ix]/scale/np.sqrt(db),'sqeuclidean')
        gh=cdist(h[ix]/np.sqrt(hb),sc.rh[ix]/np.sqrt(hb),'sqeuclidean')
        for robust in (False,True):
            heat=robust_heat_squared(gh) if robust else gh;c=gd+heat;ii,jj=linear_sum_assignment(c)
            assert len(ii)==len(ix) and len(set(jj))==len(ix)
            dc=gd[ii,jj];hc=heat[ii,jj];total=dc+hc
            lower=np.sum(((d[ix].mean(0)-sc.rd[ix].mean(0))/scale)**2)/db
            assert total.mean()+1e-10>=lower
            allparts[robust].append(dict(type=int(typ),degree=float(dc.mean()),heat=float(hc.mean()),total=float(total.mean())))
            if robust:
                for j,i in enumerate(ix):nodes.append(dict(type=int(typ),node=int(i),matched_reference=int(ix[jj[j]]),degree=float(d[i].sum()),heat_cost=float(hc[j]),degree_cost=float(dc[j]),contribution=float(total[j]/len(ix)/len(sc.scales))))
    return {('robust' if k else 'original'):dict(total=float(np.mean([r['total'] for r in v])),degree=float(np.mean([r['degree'] for r in v])),heat=float(np.mean([r['heat'] for r in v])),parts=v) for k,v in allparts.items()},nodes
