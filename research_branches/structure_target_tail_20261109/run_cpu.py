"""One frozen tail-constrained target on the archived finite graph universe."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,itertools,json,resource,signal,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/structure_target_constraints_20261109'
OUT=ROOT/'results/structure_target_tail_20261109'
START=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(x[0]));w.writeheader();w.writerows(x)
def entropy(p):
    keep=p>0
    return float(-p[keep]@np.log(p[keep]))
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='CPU finite target calculation only; old 5-node graph universe, reference-derived support and four old errors unchanged',
        normalized_error='L=max_j phi_j/epsilon_j; epsilon fixed in previous experiment from 9 legal reference toggles',
        constraints='Retain four E(phi_j)<=epsilon_j; additionally P(L>1)<=0.10 and E[(L-1)+]<=0.10',
        design='Both 0.10 limits fixed engineering examples, not learned thresholds or statistical confidence. One candidate; no grid or final quality selection.',
        objective='min KL(P||mu); mu and previous target distributions reused unchanged',
        numerical='L>1+1e-12 for floating threshold membership only; original optimization residual tolerance 1e-7',
        outputs='Exact expectations, per-graph tails, same-type isomorphism diversity, identical/isomorphic pair risk, global diversity bound, solver diagnostics',
        limits=dict(seconds=120,threads=1,memory_MiB=1024,GPU=0),
        forbidden='No network, training, random graph sampling, new teacher chain, real-graph scoring, or production changes'))
    hashes={}
    manifest=json.loads((OLD/'completion.json').read_text())
    def bind(p):
        p=Path(p);h=sha(p)
        if str(p) in manifest['output_hashes']:assert h==manifest['output_hashes'][str(p)]
        hashes[str(p)]=h
        return p
    z=np.load(bind(OLD/'completed/targets.npz'))
    bounds=json.loads(bind(OLD/'completed/frozen_bounds.json').read_text())
    univ=ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz'
    previous_binding=json.loads(bind(OLD/'completed/binding.json').read_text())
    assert sha(univ)==previous_binding['assets'][str(univ)]
    u=np.load(bind(univ));codes=z['codes'];F=z['features'];eps=z['bounds'];canonical=z['canonical'];mu=z['sparse_base'];B=u['bits'][codes];q=u['query'];types=u['types'];fam=u['families']
    assert np.array_equal(eps,np.array(bounds['bounds'])) and np.all(mu>0)
    r=F/eps;L=r.max(1);bad=L>1+1e-12;excess=np.maximum(0,L-1)
    A=np.column_stack([r,bad.astype(float),excess]);limits=np.array([1.,1.,1.,1.,.1,.1]);logmu=np.log(mu)
    save('frozen_inputs',dict(old_bounds=eps.tolist(),constraint_limits=limits.tolist(),
        feature_names=['number_ratio','joint_degree_ratio','triangle_ratio','component_ratio','bad_indicator','excess'],
        graph_count=len(codes),good_graphs=int((~bad).sum()),bad_graphs=int(bad.sum()),
        good_classes=len(set(canonical[~bad])),bad_classes=len(set(canonical[bad]))))
    def obj(lam):
        w=logmu-A@lam;lz=logsumexp(w);p=np.exp(w-lz)
        return float(lz+lam@limits),limits-p@A
    runs=[];probs=[]
    for init in [np.zeros(6),np.ones(6)]:
        ans=minimize(obj,init,jac=True,method='L-BFGS-B',bounds=[(0,None)]*6,
            options=dict(ftol=1e-15,gtol=1e-10,maxiter=2000,maxls=50))
        lam=ans.x;value,g=obj(lam);p=np.exp(logmu-A@lam-logsumexp(logmu-A@lam))
        residual=float(abs(np.where(lam>1e-7,g,np.minimum(g,0))).max())
        primal=float(p@(np.log(p)-logmu));gap=primal+value
        runs.append(dict(success=bool(ans.success),message=str(ans.message),iterations=int(ans.nit),multipliers=lam.tolist(),
            expectations=(p@A).tolist(),kkt_residual=residual,max_violation=float(max(0,np.max(p@A-limits))),
            primal_KL=primal,dual_bound=-value,gap=gap))
        probs.append(p)
        assert ans.success and residual<1e-7 and np.max(p@A-limits)<1e-7 and abs(gap)<1e-6
    p=probs[0];lam=np.array(runs[0]['multipliers'])
    assert abs(probs[0]-probs[1]).max()<1e-7
    centered=A-p@A;H=centered.T@(p[:,None]*centered)
    fd=np.column_stack([(obj(lam+np.eye(6)[j]*1e-5)[1]-obj(lam-np.eye(6)[j]*1e-5)[1])/2e-5 for j in range(6)])
    assert abs(H-fd).max()<1e-7
    save('solver',dict(runs=runs,probability_difference=float(abs(probs[0]-probs[1]).max()),hessian_error=float(abs(H-fd).max())))
    # Independent graph-level quantities, no stochastic samples.
    adj=np.zeros((len(codes),5,5),bool);adj[:,q[0],q[1]]=B;adj[:,q[1],q[0]]=B
    triangles=np.zeros(len(codes),int);comp=[]
    for i,j,k in itertools.combinations(range(5),3):triangles+=adj[:,i,j]&adj[:,i,k]&adj[:,j,k]
    for a in adj:
        reach=a|np.eye(5,dtype=bool)
        for k in range(5):reach|=reach[:,k,None]&reach[None,k,:]
        comp.append(len({tuple(row) for row in reach}))
    comp=np.array(comp);refcode=bounds['reference'];refidx=int(np.flatnonzero(codes==refcode)[0]);refclass=canonical[refidx]
    targets={n:z[n] for n in ['sparse_base','unit_energy','bounded_expected_error','zero_error_boundary']};targets['tail_constrained']=p
    def top_mean(weights,values,fraction):
        order=np.argsort(-values,kind='stable');left=fraction;value=0.
        for i in order:
            w=min(left,float(weights[i]));value+=w*values[i];left-=w
            if left<=1e-14:break
        assert left<1e-12
        return float(value/fraction)
    rows=[];details=[]
    for name,w in targets.items():
        cp=np.bincount(canonical,weights=w,minlength=1024);beta=float(w@bad)
        order=np.argsort(L);cum=np.cumsum(w[order])
        row=dict(target=name,bad_probability=beta,mean_excess=float(w@excess),
            mean_excess_given_bad=None if beta==0 else float(w@excess/beta),
            worst_ten_percent_mean_ratio=top_mean(w,L,.1),
            ratio_q90=float(L[order[min(np.searchsorted(cum,.90),len(cum)-1)]]),
            ratio_q99=float(L[order[min(np.searchsorted(cum,.99),len(cum)-1)]]),
            probability_ratio_over2=float(w@(L>2+1e-12)),probability_ratio_over3=float(w@(L>3+1e-12)),
            expected_edges=float(w@B.sum(1)),expected_triangles=float(w@triangles),
            triangle_count_MAE=float(w@abs(triangles-triangles[refidx])),disconnected_probability=float(w@(comp>1)),
            effective_labeled_graphs=float(np.exp(entropy(w))),effective_isomorphism_classes=float(np.exp(entropy(cp))),
            identical_pair_probability=float(w@w),isomorphic_pair_probability=float(cp@cp),
            reference_isomorphism_mass=float(w@(canonical==refclass)),KL_to_base=float(w[w>0]@(np.log(w[w>0])-logmu[w>0])))
        for j,n in enumerate(['number','joint_degree','typed_triangle','components']):
            row[n+'_mean_error']=float(w@F[:,j]);row[n+'_exceedance']=float(w@(r[:,j]>1+1e-12))
        rows.append(row)
        for i,c in enumerate(codes):details.append(dict(target=name,graph_code=int(c),probability=float(w[i]),
            max_error_ratio=float(L[i]),bad=bool(bad[i]),excess=float(excess[i]),isomorphism_class=int(canonical[i])))
    save('summary',rows);save('graph_probabilities',details)
    # All classes are entirely good or bad by type-invariant metrics.
    good_classes=set(canonical[~bad]);bad_classes=set(canonical[bad]);assert not(good_classes&bad_classes)
    G=len(good_classes);D=len(bad_classes);beta=.1
    # With beta <= .1, this entropy bound increases over the allowed interval here.
    assert beta<D/(G+D)
    h=-beta*np.log(beta)-(1-beta)*np.log(1-beta)
    entropy_cap=float(h+(1-beta)*np.log(G)+beta*np.log(D))
    collision_floor=float((1-beta)**2/G+beta**2/D)
    assert np.exp(entropy(np.bincount(canonical,weights=p,minlength=1024)))<=np.exp(entropy_cap)+1e-6
    assert p@excess<=.1+1e-7 and top_mean(p,L,.1)<=2+1e-6
    # Lower-bound KL cost of enforcing good mass, by coarse-graining data processing.
    qbad=float(mu@bad)
    klfloor=float(beta*np.log(beta/qbad)+(1-beta)*np.log((1-beta)/(1-qbad)))
    save('global_bounds',dict(good_classes=G,bad_classes=D,maximum_bad_mass=beta,
        effective_isomorphism_class_upper_bound=float(np.exp(entropy_cap)),
        isomorphic_pair_probability_lower_bound=collision_floor,KL_to_base_lower_bound_from_bad_mass=klfloor,
        worst_ten_percent_mean_ratio_upper_bound=2.,
        caveat='Bounds hold for ANY distribution satisfying this frozen good/bad partition and bad-mass limit, not only this solver. Finite universe only.'))
    np.savez_compressed(OUT/'targets.npz',codes=codes,features=F,old_bounds=eps,L=L,canonical=canonical,**targets)
    assert all(sha(f)==s for f,s in hashes.items())
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets=hashes))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        GPU_hours=0,model_calls=0,training_updates=0,new_random_graphs=0,old_assets_unchanged=True,formal_training_authorized=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
