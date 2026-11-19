"""Exact conditional expectation of finite-frequency joint-degree hard loss.

Independent undirected Bernoulli slots conditional on a fixed model input.
Same-type nodes share their connecting edge; their covariance is NOT omitted.
The fixed Fourier feature set is historical, not a complete graph descriptor.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
torch.set_num_threads(1)

def risks(p,types,q,f,frozen):
    """p is CPU float64 tensor; return type-equal scalar risks and parts."""
    assert p.dtype==torch.float64 and p.device.type=='cpu'
    n=len(types);qt=torch.as_tensor(q,dtype=torch.long);ft=torch.as_tensor(f,dtype=torch.long)
    d=torch.zeros((n,len(frozen['payload']['profile']['family_pairs'])),dtype=p.dtype)
    flat=d.flatten();index=qt* d.shape[1]+ft[None,:]
    d=flat.index_add(0,index.flatten(),p.repeat(2)).reshape_as(d)
    parts=[]
    for typ in sorted(np.unique(types)):
        c=str(typ);w=torch.as_tensor(frozen['payload']['omega'][c],dtype=p.dtype)
        tar=frozen['payload']['target_cf'][c]
        target=torch.as_tensor(np.array(tar['real'])+1j*np.array(tar['imag']),dtype=torch.complex128)
        h=torch.expm1(1j*w[:,ft].T);factor=1+p[:,None]*h
        assert torch.min(factor.real)>.87
        logs=torch.log(factor)
        accum=torch.zeros((n,len(w)),dtype=torch.complex128)
        accum=accum.index_add(0,qt[0],logs).index_add(0,qt[1],logs)
        phi=torch.exp(accum);ix=torch.as_tensor(np.flatnonzero(types==typ));nt=len(ix)
        mu=phi[ix].mean(0);bias=abs(mu-target).square()
        diag=(1-abs(phi[ix]).square()).sum(0)
        internal=np.flatnonzero((types[q[0]]==typ)&(types[q[1]]==typ))
        if len(internal):
            e=torch.as_tensor(internal);u=qt[0,e];v=qt[1,e]
            excluding=phi[u]*phi[v].conj()/abs(factor[e]).square()
            cov=(excluding*(p[e]*(1-p[e]))[:,None]*abs(h[e]).square()).real.sum(0)*2
        else:cov=torch.zeros_like(diag)
        variance=(diag+cov)/(nt*nt)
        plug=abs(torch.exp(1j*(d[ix]@w.T)).mean(0)-target).square()
        parts.append(dict(type=int(typ),bias=bias.mean(),variance=variance.mean(),diagonal_variance=(diag/(nt*nt)).mean(),
                          covariance=(cov/(nt*nt)).mean(),risk=(bias+variance).mean(),plugin=plug.mean()))
    return {key:torch.stack([r[key] for r in parts]).mean() for key in ('risk','bias','variance','diagonal_variance','covariance','plugin')},parts

def hard_losses(bits,types,q,f,frozen):
    """Independent hard evaluation using degree counts, no probability formula."""
    from scipy.sparse import coo_matrix
    B=len(bits);n=len(types);r=len(frozen['payload']['profile']['family_pairs']);E=len(f)
    incidence=coo_matrix((np.ones(2*E),(np.tile(np.arange(E),2),(q*r+f[None,:]).ravel())),shape=(E,n*r)).tocsr()
    degree=np.asarray((incidence.T@np.asarray(bits,float).T).T).reshape(B,n,r)
    allparts=[]
    for typ in sorted(np.unique(types)):
        w=np.array(frozen['payload']['omega'][str(typ)]);t=frozen['payload']['target_cf'][str(typ)]
        target=np.array(t['real'])+1j*np.array(t['imag'])
        phi=np.exp(1j*(degree[:,types==typ]@w.T)).mean(1)
        allparts.append(np.mean(abs(phi-target)**2,axis=1))
    return np.mean(allparts,axis=0)

def probability_metrics(p,types,q,f,ref,relations):
    n=len(types);P=np.zeros((n,n));P[q[0],q[1]]=p;P+=P.T
    d=np.zeros((n,relations));rd=np.zeros_like(d)
    for endpoint in q:
        np.add.at(d,(endpoint,f),p);np.add.at(rd,(endpoint,f),ref)
    # Exact Poisson-binomial total-degree PMF, including p=0 and p=1 safely.
    pmf=np.zeros((n,n+1));pmf[:,0]=1
    for j in range(n):
        b=P[:,j,None];old=pmf.copy();pmf=old*(1-b);pmf[:,1:]+=old[:,:-1]*b
    assert np.max(abs(pmf.sum(1)-1))<1e-10
    groups=[]
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    for typ in sorted(np.unique(types)):
        ix=types==typ;rr=rd[ix];active=rr.sum(0)>0;total=rr.sum(1)
        threshold=max(2,int(np.ceil(np.quantile(total,.9))))
        C=cdist(d[ix][:,active],rr[:,active],metric='cityblock')/active.sum();a,b=linear_sum_assignment(C)
        groups.append(dict(type=int(typ),nodes=int(ix.sum()),expected_low_fraction=float(pmf[ix,:2].sum(1).mean()),
            reference_low_fraction=float((total<=1).mean()),hub_threshold=threshold,expected_hub_fraction=float(pmf[ix,threshold:].sum(1).mean()),
            reference_hub_fraction=float((total>=threshold).mean()),expected_degree_joint_L1_proxy=float(C[a,b].mean())))
    return dict(expected_edges=float(p.sum()),expected_triangles=float(np.trace(P@P@P)/6),
                relation_expected_edges=np.bincount(f,weights=p,minlength=relations).tolist(),
                low_fraction_error=float(np.mean([abs(g['expected_low_fraction']-g['reference_low_fraction']) for g in groups])),
                hub_fraction_error=float(np.mean([abs(g['expected_hub_fraction']-g['reference_hub_fraction']) for g in groups])),
                joint_expected_degree_proxy=float(np.mean([g['expected_degree_joint_L1_proxy'] for g in groups])),types=groups)
