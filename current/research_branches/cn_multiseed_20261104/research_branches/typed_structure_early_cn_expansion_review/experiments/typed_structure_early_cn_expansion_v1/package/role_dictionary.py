"""Frozen protocol: node-held-out, feature-group-held-out rank selection.
No neural prediction, generation output or role labels enter this module.
"""
import hashlib,json,time,warnings
import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

NMF_OPTIONS=dict(init='nndsvda',solver='cd',beta_loss='frobenius',tol=1e-5,
                 max_iter=50000,shuffle=False,alpha_W=0.,alpha_H=0.,l1_ratio=0.)

def seed32(root,**labels):
    payload=json.dumps(dict(root=int(root),**labels),sort_keys=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4],'little')

def feature_groups(relations=10):
    base=3*relations
    groups=[[j]+[base+r*2*base+op*base+j for r in range(relations) for op in range(2)] for j in range(base)]
    assert sorted(sum(groups,[]))==list(range(base*(1+2*relations)))
    perm=np.random.default_rng(2026102853).permutation(base)
    sides=[np.array(sorted(sum([groups[j] for j in part],[])),dtype=np.int64) for part in [perm[:base//2],perm[base//2:]]]
    return groups,sides,perm

def node_folds(types,dataset):
    result={}
    for typ in range(int(types.max())+1):
        ids=np.flatnonzero(types==typ);seed=seed32(2026102851,dataset=dataset,type=typ,purpose='node_folds');p=np.random.default_rng(seed).permutation(ids)
        result[typ]=dict(seed=seed,folds=[p[:len(ids)//2],p[len(ids)//2:]],Kmax=min(8,len(ids)//2,len(ids)-len(ids)//2))
    return result

def weighted_rows(features,nodes):
    if len(features)!=7:raise ValueError('one clean and six noisy states required')
    return np.concatenate([f[nodes]*(np.sqrt(2.) if i==0 else 1.) for i,f in enumerate(features)],axis=0)

def fit(matrix,k,seed,artifact=None):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');model=NMF(n_components=k,random_state=seed,**NMF_OPTIONS);weights=model.fit_transform(matrix)
    basis=model.components_.copy();norm=np.linalg.norm(basis,axis=1)
    finite=bool(np.isfinite(basis).all() and np.isfinite(norm).all() and np.isfinite(model.reconstruction_err_));positive=bool((norm>0).all())
    converged=model.n_iter_<NMF_OPTIONS['max_iter'] and not any(issubclass(w.category,ConvergenceWarning) for w in ws)
    receipt=dict(k=k,seed=seed,seconds=time.perf_counter()-start,iterations=int(model.n_iter_),converged=converged,finite=finite,positive_basis_rows=positive,
                 reconstruction_error=float(model.reconstruction_err_) if finite else None,warnings=[str(w.message) for w in ws],options=NMF_OPTIONS)
    normalized=basis/norm[:,None] if finite and positive else np.full_like(basis,np.nan)
    if artifact is not None:
        np.savez_compressed(artifact,W=weights,H=basis,normalized_basis=normalized,input_sha256=np.array(hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()),input_shape=np.array(matrix.shape),seed=np.array(seed))
        receipt['artifact_sha256']=hashlib.sha256(__import__('pathlib').Path(artifact).read_bytes()).hexdigest()
    if not finite or not positive or not converged:return None,receipt
    return basis/norm[:,None],receipt

def nnls_encode(features,basis):
    f=np.asarray(features,dtype=np.float64);h=np.asarray(basis,dtype=np.float64)
    if f.ndim!=2 or h.ndim!=2 or f.shape[1]!=h.shape[1] or h.shape[0]<1:raise ValueError('NNLS dimensions')
    if not np.isfinite(f).all() or not np.isfinite(h).all() or (f<0).any() or (h<0).any():raise FloatingPointError('invalid NNLS inputs')
    if not len(f):return np.zeros((0,len(h))),dict(kkt_max=0.,maxiter=max(240,30*len(h)))
    c=np.vstack([nnls(h.T,row,maxiter=max(240,30*len(h)))[0] for row in f]);grad=(c@h-f)@h.T
    kkt=max(float(np.maximum(-grad,0).max()),float(np.abs(c*grad).max()))
    if not np.isfinite(c).all() or kkt>1e-7:raise FloatingPointError('NNLS KKT failure: '+str(kkt))
    return c,dict(kkt_max=kkt,maxiter=max(240,30*len(h)),coefficient_sum_range=[float(c.sum(1).min()),float(c.sum(1).max())])

def masked_score(features,nodes,basis,sides):
    rows=[]
    for observed,target in [sides,sides[::-1]]:
        errors=[];max_kkt=0.
        for f in features:
            # Target columns NEVER enter coefficient fitting.
            c,receipt=nnls_encode(f[nodes][:,observed],basis[:,observed]);max_kkt=max(max_kkt,receipt['kkt_max'])
            errors.append(float(np.mean((c@basis[:,target]-f[nodes][:,target])**2)))
        weighted=.25*errors[0]+.125*sum(errors[1:])
        rows.append(dict(state_mse=errors,weighted_mse=weighted,kkt_max=max_kkt))
    return rows

def choose(scores):
    if not scores or any(not r['valid'] for r in scores):raise ValueError('incomplete candidates: no K selection permitted')
    minimum=min(r['mean_mse'] for r in scores)
    return min(r['K'] for r in scores if r['mean_mse']-minimum<=1e-12)

def encode_typed(features,matrix,offsets,types):
    matrix=np.asarray(matrix,dtype=np.float64);offsets=np.asarray(offsets,dtype=np.int64);types=np.asarray(types,dtype=np.int64)
    if offsets.ndim!=1 or offsets[0]!=0 or offsets[-1]!=len(matrix) or (np.diff(offsets)<=0).any():raise ValueError('role offsets')
    if types.shape!=(len(features),) or (types<0).any() or (types>=len(offsets)-1).any():raise ValueError('types')
    out=np.zeros((len(features),len(matrix)),dtype=np.float64);receipts={}
    for typ,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        ids=np.flatnonzero(types==typ);c,rec=nnls_encode(features[ids],matrix[lo:hi]);out[ids,lo:hi]=c;receipts[str(typ)]=rec
    return out,dict(per_type=receipts,kkt_max=max(r['kkt_max'] for r in receipts.values()),offsets=offsets.tolist(),layout='type-major variable K, other type columns zero; no row normalization')
