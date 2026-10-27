"""Fixed typed ReFeX-inspired features and frozen-basis RolX-style encoding.

Only explicit complete Xt edges enter extraction. No query/labels/node IDs.
Not an implementation of adaptive ReFeX pruning or RolX MDL rank selection.
"""
import hashlib
import json
import time
import warnings
import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

SPEC = dict(version='typed_refex_2_v1',depth=2,base='relation degree, closed-egonet internal and boundary edges',
            normalization='log1p(base count)/log1p(schema bound); recursive mean and sum/(N-1)',
            rank=8,basis_seed=2026102708,nmf_max_iter=500,nmf_tol=1e-5,
            encoding='per-node NNLS, fixed unit-L2 basis rows; no coefficient normalization',nnls_max_iter=240)

def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def extract(types, edges, families, pairs):
    types=np.asarray(types,dtype=np.int64);edges=np.asarray(edges,dtype=np.int64);families=np.asarray(families,dtype=np.int64)
    n=len(types);r=len(pairs)
    if edges.shape!=(2,len(families)) or n<2:raise ValueError('shape')
    if ((edges<0)|(edges>=n)).any() or (edges[0]>=edges[1]).any():raise ValueError('canonical endpoints required')
    if ((families<0)|(families>=r)).any() or len(np.unique(edges[0]*n+edges[1]))!=len(families):raise ValueError('families/duplicates')
    if not np.array_equal(np.sort(types[edges],axis=0).T,np.asarray(pairs)[families]):raise ValueError('schema')
    u,v=edges;adj=csr_matrix((np.ones(2*len(u)),(np.r_[u,v],np.r_[v,u])),shape=(n,n))
    deg=np.zeros((n,r));np.add.at(deg,(u,families),1);np.add.at(deg,(v,families),1)
    # Independent closed egonets; O(sum_i E + sum_i degree_i), bounded N*E work.
    # No candidate-edge loop and no N*N*N tensor. Dense boolean membership is N only.
    inside=np.zeros_like(deg);boundary=np.zeros_like(deg)
    for i in range(n):
        member=np.zeros(n,dtype=bool);member[i]=True;member[adj.indices[adj.indptr[i]:adj.indptr[i+1]]]=True
        a,b=member[u],member[v]
        inside[i]=np.bincount(families[a&b],minlength=r)
        boundary[i]=np.bincount(families[a^b],minlength=r)
    sizes=np.bincount(types,minlength=int(types.max())+1)
    cap=np.array([sizes[a]*(sizes[a]-1)//2 if a==b else sizes[a]*sizes[b] for a,b in pairs])
    bounds=np.r_[np.full(r,n-1),cap,cap]
    base=np.divide(np.log1p(np.c_[deg,inside,boundary]),np.log1p(bounds),
                   out=np.zeros((n,3*r)),where=bounds>0)
    all_features=[base];prev=base;d=np.asarray(adj.sum(1))
    for _ in range(SPEC['depth']):
        sums=adj@prev;mean=np.divide(sums,d,out=np.zeros_like(sums),where=d>0)
        prev=np.c_[mean,sums/(n-1)];all_features.append(prev)
    value=np.concatenate(all_features,axis=1)
    if not np.isfinite(value).all() or (value<0).any():raise FloatingPointError('ReFeX')
    return value

def fit_basis(features):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        model=NMF(n_components=SPEC['rank'],init='nndsvda',solver='cd',beta_loss='frobenius',
                  tol=SPEC['nmf_tol'],max_iter=SPEC['nmf_max_iter'],random_state=SPEC['basis_seed'])
        model.fit_transform(features)
    basis=model.components_.copy();norm=np.linalg.norm(basis,axis=1)
    if (norm==0).any():raise ValueError('zero basis row; no rank replacement')
    basis/=norm[:,None]
    return basis,dict(seconds=time.perf_counter()-start,iterations=int(model.n_iter_),
                      warnings=[str(w.message) for w in ws],reconstruction_error=float(model.reconstruction_err_),
                      rank=SPEC['rank'],sha256=digest(basis),source='only this dataset training reference features')

def encode(features,basis):
    basis=np.asarray(basis,dtype=np.float64)
    if basis.shape!=(SPEC['rank'],features.shape[1]) or not np.isfinite(basis).all() or (basis<0).any():raise ValueError('basis')
    # No fitted per-node role table; each new Xt is solved independently.
    coeff=np.vstack([nnls(basis.T,row,maxiter=SPEC['nnls_max_iter'])[0] for row in features])
    grad=(coeff@basis-features)@basis.T
    kkt=max(float(np.maximum(-grad,0).max()),float(np.abs(coeff*grad).max()))
    if not np.isfinite(coeff).all() or kkt>1e-7:raise FloatingPointError('NNLS KKT failure %g'%kkt)
    return coeff,dict(kkt_max=kkt,reconstruction_l2=float(np.linalg.norm(coeff@basis-features)),
                      coefficient_sum_range=[float(coeff.sum(1).min()),float(coeff.sum(1).max())])
