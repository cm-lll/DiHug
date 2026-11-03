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

SPEC = dict(version='typed_refex_relation_one_round_v2',depth=1,base='relation degree, closed-egonet internal and boundary edges',
            normalization='log1p(base count)/log1p(schema bound); ONE round: for each relation mean and sum/(N-1) of base features',
            rank=4,basis_seed=2026102708,nmf_max_iter=500,nmf_tol=1e-5,
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
    all_features=[base]
    for relation in range(r):
        mask=families==relation;uu,vv=u[mask],v[mask]
        relation_adj=csr_matrix((np.ones(2*len(uu)),(np.r_[uu,vv],np.r_[vv,uu])),shape=(n,n))
        sums=relation_adj@base;d=np.asarray(relation_adj.sum(1))
        mean=np.divide(sums,d,out=np.zeros_like(sums),where=d>0)
        all_features.append(np.c_[mean,sums/(n-1)])
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
                      rank=SPEC['rank'],sha256=digest(basis),source='caller-declared fitting material; v2 ONLY synthetic engineering')

def encode(features,basis,types=None):
    basis=np.asarray(basis,dtype=np.float64)
    if basis.ndim==3:
        if types is None or basis.shape[1:]!=(4,features.shape[1]):raise ValueError('typed basis shape/types')
        types=np.asarray(types,dtype=np.int64)
        if types.shape!=(len(features),) or (types<0).any() or (types>=len(basis)).any():raise ValueError('types')
        coeff=np.zeros((len(features),4*len(basis)));receipts={}
        for typ in range(len(basis)):
            ids=np.flatnonzero(types==typ)
            if not len(ids):continue
            c,receipt=encode(features[ids],basis[typ]);coeff[ids,4*typ:4*(typ+1)]=c;receipts[str(typ)]=receipt
        return coeff,dict(per_type=receipts,kkt_max=max(v['kkt_max'] for v in receipts.values()),layout='type-major 4 coefficients, all other type columns zero; no row normalization')
    if basis.shape!=(SPEC['rank'],features.shape[1]) or not np.isfinite(basis).all() or (basis<0).any():raise ValueError('basis')
    # No fitted per-node role table; each new Xt is solved independently.
    coeff=np.vstack([nnls(basis.T,row,maxiter=SPEC['nnls_max_iter'])[0] for row in features])
    grad=(coeff@basis-features)@basis.T
    kkt=max(float(np.maximum(-grad,0).max()),float(np.abs(coeff*grad).max()))
    if not np.isfinite(coeff).all() or kkt>1e-7:raise FloatingPointError('NNLS KKT failure %g'%kkt)
    return coeff,dict(kkt_max=kkt,reconstruction_l2=float(np.linalg.norm(coeff@basis-features)),
                      coefficient_sum_range=[float(coeff.sum(1).min()),float(coeff.sum(1).max())])

def feature_width(relations):return 3*relations*(1+2*relations)

def synthetic_bases(n_types,n_families):
    """No dataset feature values, clean edges or original IDs are accepted."""
    rng=np.random.default_rng(SPEC['basis_seed']);bases=[];receipts=[]
    for typ in range(n_types):
        # Separate rank-4 synthetic nonnegative matrices for each type.
        factors=rng.uniform(.1,1,(4,feature_width(n_families)))
        coefficients=rng.uniform(.1,1,(24,4))
        basis,receipt=fit_basis(coefficients@factors)
        bases.append(basis);receipts.append(receipt)
    return np.stack(bases),dict(purpose='SYNTHETIC ENGINEERING ONLY; no reference graph fit; not formal dictionary',per_type=receipts)
