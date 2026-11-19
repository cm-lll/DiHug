"""Finite target-design audit. No network, sampling, or production changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, itertools, json, resource, signal, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/structure_target_constraints_20261109/completed'
ASSET=ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz'
START=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='Exact old 5-node universe. Target-distribution audit only; no new random graphs or teacher chains.',
        base='Independent relation Bernoulli with reference counts; exact 0/1 support. NOT previous synthetic noise priors; no reverse-kernel claims.',
        constraints='Four unchanged stored toy errors: relation counts, anonymous joint degree, typed triangles, components. Bounds = arithmetic mean over ALL support-legal single toggles of reference. One predefined illustrative tolerance, not adoption or search.',
        target='Minimize KL(P||mu) subject to expected errors <= frozen bounds. Dual nonnegative multipliers solved; no temperature grid.',
        references=['mu','mu*exp(-mean(errors))','constrained KL projection','mu conditioned on all errors exactly zero'],
        verification='Dual KKT/gap, deterministic second solver initialization, finite-difference Hessian, independent errors, type-isomorphism classes.',
        limits=dict(seconds=120,threads=1,memory_MiB=1024,GPU=0),
        exclusions='No real-graph new scoring, training, model, generation, or formal target selection.'))
    asset_hash=sha(ASSET)
    z=np.load(ASSET);bits=z['bits'];q=z['query'];fam=z['families'];types=z['types'];E=z['energy_parts']
    ref=int(z['reference']@(1<<np.arange(bits.shape[1])))
    counts=np.stack([bits[:,fam==r].sum(1) for r in range(3)],1)
    pi=counts[ref]/np.bincount(fam)
    support=np.all((pi[fam]>0)|(bits==0),1)&np.all((pi[fam]<1)|(bits==1),1)
    codes=np.flatnonzero(support);B=bits[support];F=E[support]
    mu=np.prod(np.where(B,pi[fam],1-pi[fam]),1);assert abs(mu.sum()-1)<1e-14
    logmu=np.log(mu)
    toggle_ids=[ref^(1<<int(e)) for e in np.flatnonzero((pi[fam]>0)&(pi[fam]<1))]
    eps=E[toggle_ids].mean(0)
    save('frozen_bounds',dict(reference=ref,relation_priors=pi.tolist(),support_graphs=len(codes),
        legal_reference_toggles=toggle_ids,bounds=eps.tolist(),
        names=['relation_count','joint_degree','typed_triangle','components'],
        caution='Expected-error bound in units of one-toggle shell average; not per-graph edit radius, confidence interval, universal tolerance or hard guarantee.'))
    def objective(lam):
        w=logmu-F@lam;lnZ=logsumexp(w);p=np.exp(w-lnZ)
        return float(lnZ+lam@eps),eps-p@F
    solutions=[]
    for init in [np.zeros(4),np.ones(4)]:
        ans=minimize(objective,init,jac=True,bounds=[(0,None)]*4,method='L-BFGS-B',
            options={'ftol':1e-15,'gtol':1e-10,'maxiter':2000,'maxls':50})
        lam=ans.x;val,grad=objective(lam);lnZ=logsumexp(logmu-F@lam);p=np.exp(logmu-F@lam-lnZ)
        residual=float(np.max(abs(np.where(lam>1e-7,grad,np.minimum(grad,0)))))
        primal=float(p@(np.log(p)-logmu));dual=-val
        solutions.append(dict(success=bool(ans.success),message=str(ans.message),iterations=int(ans.nit),
            multipliers=lam.tolist(),expected_errors=(p@F).tolist(),kkt_residual=residual,
            max_violation=float(max(0,np.max(p@F-eps))),primal_KL=primal,dual_bound=dual,gap=primal-dual))
        assert ans.success and residual<1e-7 and np.max(p@F-eps)<1e-7 and abs(primal-dual)<1e-6
    # First fixed solution is used; second only checks numerical consistency.
    lam=np.array(solutions[0]['multipliers']);pstar=np.exp(logmu-F@lam-logsumexp(logmu-F@lam))
    lam2=np.array(solutions[1]['multipliers']);p2=np.exp(logmu-F@lam2-logsumexp(logmu-F@lam2))
    assert abs(pstar-p2).max()<1e-7
    centered=F-pstar@F;hess=centered.T@(pstar[:,None]*centered)
    fd=np.column_stack([(objective(lam+np.eye(4)[i]*1e-5)[1]-objective(lam-np.eye(4)[i]*1e-5)[1])/2e-5 for i in range(4)])
    assert abs(hess-fd).max()<1e-8
    save('solver',dict(runs=solutions,probability_difference=float(abs(pstar-p2).max()),
        hessian_fd_error=float(abs(hess-fd).max()),covariance_eigenvalues=np.linalg.eigvalsh(hess).tolist()))
    # Independent adjacency/degree/permutation recomputation of the four old errors.
    lookup={tuple(v):i for i,v in enumerate(q.T)}
    adj=np.zeros((len(bits),5,5),int);adj[:,q[0],q[1]]=bits;adj[:,q[1],q[0]]=bits
    deg=np.zeros((len(bits),5,3))
    for e,(u,v) in enumerate(q.T):deg[:,u,fam[e]]+=bits[:,e];deg[:,v,fam[e]]+=bits[:,e]
    tri=np.zeros((len(bits),4));components=[]
    for nodes in itertools.combinations(range(5),3):
        a,b,c=nodes;tri[:,types[list(nodes)].sum()]+=adj[:,a,b]*adj[:,a,c]*adj[:,b,c]
    for a in adj:
        reach=a.astype(bool)|np.eye(5,dtype=bool)
        for k in range(5):reach|=reach[:,k,None]&reach[None,k,:]
        components.append(len({tuple(row) for row in reach}))
    components=np.array(components);allocation=np.zeros(len(bits))
    for t in [0,1]:
        ids=np.flatnonzero(types==t);cols=sorted(set(fam[np.isin(q[0],ids)|np.isin(q[1],ids)]))
        dref=deg[ref][ids][:,cols];scale=np.maximum(1,dref.mean(0));vals=[]
        for order in itertools.permutations(range(len(ids))):
            vals.append((np.abs(deg[:,ids][:,:,cols]-dref[list(order)])/scale).mean((1,2)))
        allocation+=np.min(vals,axis=0)/2
    independent=np.stack([(((counts-counts[ref])/np.maximum(1,counts[ref]))**2).mean(1),allocation,
        (((tri-tri[ref])/np.maximum(1,tri[ref]))**2).mean(1),(components-components[ref])**2],1)
    assert abs(independent-E).max()<1e-12
    canonical=np.arange(len(bits))
    for aa in itertools.permutations([0,1]):
        for bb in itertools.permutations([2,3,4]):
            perm=np.array(aa+bb);dest=np.zeros_like(bits)
            for k,(u,v) in enumerate(q.T):dest[:,lookup[tuple(sorted((perm[u],perm[v])))]]=bits[:,k]
            canonical=np.minimum(canonical,dest@(1<<np.arange(10)))
    unit=np.exp(logmu-F.mean(1)-logsumexp(logmu-F.mean(1)))
    exactmask=np.all(F==0,1);exact=mu*exactmask;exact/=exact.sum()
    targets={'sparse_base':mu,'unit_energy':unit,'bounded_expected_error':pstar,'zero_error_boundary':exact}
    rows=[];allrows=[]
    for name,p in targets.items():
        keep=p>0;ent=-float(p[keep]@np.log(p[keep]));classp=np.bincount(canonical[codes],weights=p,minlength=1024);ck=classp>0
        classent=-float(classp[ck]@np.log(classp[ck]))
        m=p@B;indprod=np.prod(np.where(B,m,1-m),1)
        row=dict(target=name,expected_edges=float(p@counts[support].sum(1)),expected_triangles=float(p@tri[support].sum(1)),
            disconnected_probability=float(p@(components[support]>1)),reference_probability=float(p[codes==ref][0]),
            reference_isomorphism_probability=float(p@(canonical[codes]==canonical[ref])),
            all_bounds_per_graph_probability=float(p@np.all(F<=eps+1e-12,1)),
            entropy=ent,effective_labeled_graphs=float(np.exp(ent)),effective_type_isomorphism_classes=float(np.exp(classent)),
            pairwise_identical_probability=float(p@p),pairwise_isomorphic_probability=float(classp@classp),
            joint_vs_marginal_product_TV=float(abs(p-indprod).sum()/2),
            KL_to_base=float(p[keep]@(np.log(p[keep])-logmu[keep])))
        for j,n in enumerate(['relation_count','joint_degree','typed_triangle','components']):row[n+'_error']=float(p@F[:,j])
        rows.append(row)
        for i,c in enumerate(codes):allrows.append(dict(target=name,graph_code=int(c),probability=float(p[i]),isomorphism_class=int(canonical[c])))
    save('target_summary',rows);save('target_probabilities',allrows)
    np.savez_compressed(OUT/'targets.npz',codes=codes,features=F,bounds=eps,canonical=canonical[codes],**targets)
    save('checks',dict(independent_feature_error=float(abs(independent-E).max()),
        zero_error_graphs=int(exactmask.sum()),zero_error_isomorphism_classes=len(set(canonical[codes][exactmask])),
        old_asset_unchanged=sha(ASSET)==asset_hash))
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets={str(ASSET):asset_hash}))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,model_calls=0,
        training_updates=0,new_random_graphs=0,formal_training_authorized=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
