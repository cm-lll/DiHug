"""Bounded CPU probability-space CAGrad pilot; no neural model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,time,json,csv,resource,traceback
from pathlib import Path
import importlib.util
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize,nnls
ROOT=Path(__file__).resolve().parents[2]
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=mod('coordinate',ROOT/'research_branches/coordinate_joint_pilot_20261110/run.py')
OUT=Path(os.environ.get('PILOT_OUT',str(ROOT/'results/mean_anchored_pilot_20261110')))
def save(name,data): (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
def write(name,rows):
    if rows:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def solve(G,c=.5):
    """Primal in rank-revealed row span, verified by independent dual certificate."""
    mean=G.mean(0);scale=np.linalg.norm(mean)
    if scale==0: return np.zeros_like(mean),dict(status='zero_mean',norm_ratio=0.)
    U,s,V=np.linalg.svd(G/scale,full_matrices=False)
    rank=int(np.sum(s>s[0]*max(G.shape)*np.finfo(float).eps));V=V[:rank]
    H=(G/scale)@V.T;b=H.mean(0)
    def fun(x):return -x[-1]
    def jac(x):return np.r_[np.zeros(rank),-1.]
    def cons(x):return np.r_[H@x[:-1]-x[-1],c*c-np.sum((x[:-1]-b)**2)]
    def cj(x):return np.vstack([np.c_[H,-np.ones(len(H))],np.r_[-2*(x[:-1]-b),0.]])
    initial=np.r_[b,np.min(H@b)]
    res=minimize(fun,initial,jac=jac,method='SLSQP',constraints=dict(type='ineq',fun=cons,jac=cj),options=dict(ftol=1e-12,maxiter=1000))
    first_status=str(res.message)
    if not res.success:
        # Restart the SAME convex problem once from a strictly feasible point.
        # This changes neither c, constraints, nor certification tolerances.
        trial=b+.99*(res.x[:-1]-b)*min(1,c/max(c,np.linalg.norm(res.x[:-1]-b)))
        res=minimize(fun,np.r_[trial,np.min(H@trial)-1e-8],jac=jac,method='SLSQP',constraints=dict(type='ineq',fun=cons,jac=cj),options=dict(ftol=1e-12,maxiter=1000))
    y=res.x[:-1];v=y-b;value=float(np.min(H@y));slack=H@y-value;ball=c*c-v@v
    # Nonnegative KKT multipliers, including complementarity. No ridge.
    M=np.vstack([np.c_[H.T,-2*v],np.r_[np.ones(len(H)),0.],np.c_[np.diag(slack),np.zeros(len(H))],np.r_[np.zeros(len(H)),ball]])
    rhs=np.r_[np.zeros(rank),1.,np.zeros(len(H)+1)]
    mult,kkt=nnls(M,rhs,maxiter=10000);w=mult[:-1]
    if w.sum()<=0: raise ArithmeticError('no dual simplex certificate')
    w=w/w.sum();hw=w@H;upper=float(hw@b+c*np.linalg.norm(hw));gap=upper-value
    d=scale*(y@V);tol=2e-6*max(1,abs(value),abs(upper))
    cert=dict(first_status=first_status,status=str(res.message),success=bool(res.success),rank=rank,iterations=int(res.nit),primal=float(value),dual=upper,gap=gap,tolerance=tol,ball_violation=float(max(0,-ball)),kkt_residual=float(kkt),multipliers=w.tolist(),norm_ratio=float(np.linalg.norm(d)/scale),mean_descent_ratio=float(mean@d/(scale*scale)))
    if not (res.success and gap>=-tol and gap<=tol and ball>=-1e-8 and mean@d >= (.5-1e-7)*scale*scale): raise ArithmeticError(str(cert))
    return d,cert

def tests():
    rng=np.random.default_rng(2026111081);cases=[np.array([[1.],[-.2]]),np.ones((5,3)),np.array([[1.,0],[-1.,0]])]
    for i in range(24):
        G=rng.normal(size=(5,7))
        if i%3==0:G[1]=G[0]
        if i%3==1:G[:,3:]=0
        cases.append(G)
    certs=[]
    for G in cases:
        d,cert=solve(G);certs.append(cert)
        m=G.mean(0);norm=np.linalg.norm(m)
        assert np.linalg.norm(d-m)<=.5*norm+1e-7
        if norm:
            dirs=rng.normal(size=(200,G.shape[1]));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
            cand=m+.5*norm*dirs
            assert np.min(G@d)>=np.max(np.min(cand@G.T,axis=1))-1e-6
    np.testing.assert_allclose(solve(cases[0])[0],[.2],atol=1e-9)
    return dict(passed=True,cases=len(cases),random_feasible_comparisons=200*26,certificates=certs)

def main():
    start=time.time();OUT.mkdir(exist_ok=True);rows=[];decisions=[];risks=[];relations=[];contrasts=[];failures=[];samples=0
    receipt=dict(status='running',GPU_hours=0,neural_calls=0,network_backwards=0,optimizer_updates=0,reverse_chains=0)
    try:
        old.torch.set_num_threads(1);old.torch.set_num_interop_threads(1)
        ap=json.loads((ROOT/'results/density_arrangement_audit_20261110/protocol.json').read_text());proto=json.loads((old.PREV/'protocol.json').read_text());bindings=json.loads((ROOT/'results/coordinate_joint_pilot_20261110/bindings.json').read_text())
        for p,h in bindings.items():assert old.audit.sha(p)==h,p
        bindings[str(Path(__file__))]=old.audit.sha(__file__);save('bindings.json',bindings);save('cpu_gate.json',tests())
        for task in proto['tasks']:
            ds,seed=task['dataset'],task['seed'];source=Path(task['response_source']);records=[r for r in ap['conditions_manifest'] if r['dataset']==ds and r['seed']==seed and r['alpha']==1]
            data=np.load(task['data']);first=np.load(records[0]['before']);core=old.audit.HardScore(data['types'],first['query'],first['families'],first['labels'],len(data['schema']));exact=old.exactmod.ExactHardMoments(core);moment=old.audit.Moments(core)
            for rec in records:
                if time.time()-start>600: raise TimeoutError('science stop at 600 seconds')
                meta={k:rec[k] for k in ('dataset','seed','panel','t')};label=rec['panel']
                try:
                    p0=np.load(rec['before'])['probability'];p1=np.load(rec['after'])['probability'];active,z,delta,_,_=old.decompose(p0,p1,core.f)
                    bc=np.load(source/f'{label}_before_scored.npz');ac=np.load(source/f'{label}_after_scored.npz');j=ap['panels'].index(label)-1
                    es=seed+(proto['same_input_evaluation_seed_offset'] if label=='same' else proto['evaluation_hard_seed_offset']+j);u=np.random.default_rng(es).random((256,len(p0)))
                    G=np.zeros((5,len(p0)));v,g,_=exact.value_gradient(p0);G[[0,2,3]]=g;G[[1,4]]=old.loo(p0,(u[:128]<p0).astype(np.uint8),bc['costs_remaining'][:128]);G[:,~active]=0
                    np.testing.assert_allclose(v,bc['values'][[0,2,3]],atol=1e-11,rtol=1e-9)
                    d,cert=solve(G);mean=G.mean(0);base_norm=np.linalg.norm(mean);orig_norm=np.linalg.norm(delta)
                    if not base_norm or not orig_norm: raise ArithmeticError('zero scale: uninformative condition')
                    eta=orig_norm/base_norm;dc=-eta*d;dm=-eta*mean;matched=dm*(np.linalg.norm(dc)/np.linalg.norm(dm))
                    vectors={'average':dm,'cagrad':dc,'same_norm_average':matched};probs={'unchanged':p0,'original':p1};actual={}
                    for arm,vec in vectors.items():
                        p=p0.copy();p[active]=expit(z[active]+vec[active]);assert np.isfinite(p).all() and ((p[active]>0)&(p[active]<1)).all()
                        probs[arm]=p;real=np.zeros_like(vec);real[active]=np.log(p[active])-np.log1p(-p[active])-z[active];actual[arm]=real
                        np.testing.assert_allclose(real,vec,atol=1e-9,rtol=1e-7)
                    assert abs(np.linalg.norm(actual['cagrad'])-np.linalg.norm(actual['same_norm_average']))<1e-7*max(1,orig_norm)
                    vals={}
                    for arm,p in probs.items():
                        cached=bc if arm=='unchanged' else ac if arm=='original' else None
                        if cached is not None: value=cached['values'].copy();cost=cached['costs_remaining'][128:]
                        else:
                            value=np.zeros(5);value[[0,2,3]]=moment.score(p)[0];cost=old.remaining(core,(u[128:]<p).astype(np.uint8));samples+=128
                        value[[1,4]]=cost.mean(0);vals[arm]=(value,cost)
                    before=vals['unchanged'][0]
                    for arm,(value,cost) in vals.items():
                        rows.append(dict(**meta,arm=arm,**dict(zip(core.names,map(float,value))),mean_five=float(value.mean()),expected_edges=float(probs[arm].sum())))
                        for k,n in enumerate(core.names):
                            se=float((cost-vals['unchanged'][1])[:,[1,4].index(k)].std(ddof=1)/np.sqrt(128)) if k in (1,4) else 0.
                            risks.append(dict(**meta,arm=arm,metric=n,value=float(value[k]),change=float(value[k]-before[k]),paired_MC_SE=se))
                        for f in range(core.r):
                            pos=(core.f==f)&(core.reference==1);neg=(core.f==f)&(core.reference==0);p=probs[arm]
                            relations.append(dict(**meta,arm=arm,family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),expected_edges=float(p[core.f==f].sum()),missed_mass=float((1-p[pos]).sum()),false_mass=float(p[neg].sum())))
                    for arm in ('average','same_norm_average','original','unchanged'):
                        dif=vals['cagrad'][0]-vals[arm][0];samplediff=vals['cagrad'][1]-vals[arm][1]
                        contrasts.append(dict(**meta,control=arm,**dict(zip(core.names,map(float,dif))),mean_five=float(dif.mean()),mean_paired_MC_SE=float((samplediff.sum(1)/5).std(ddof=1)/np.sqrt(128))))
                    decisions.append(dict(**meta,eta=float(eta),original_norm=float(orig_norm),candidate_norm=float(np.linalg.norm(dc)),norm_ratio=cert['norm_ratio'],mean_descent_ratio=cert['mean_descent_ratio'],cosine_to_average=float(d@mean/(np.linalg.norm(d)*base_norm)),solver_gap=cert['gap'],solver_tolerance=cert['tolerance'],max_first_order_change=float((G@dc).max()),finite_mean_change=float(vals['cagrad'][0].mean()-before.mean())))
                    np.savez_compressed(OUT/f'{ds}_{seed}_{label}.npz',gradients=G,**{'p_'+k:p for k,p in probs.items()},values=np.stack([x[0] for x in vals.values()]),arms=np.array(list(vals)),evaluation_remaining_costs=np.stack([x[1] for x in vals.values()]))
                    save(f'{ds}_{seed}_{label}_solver.json',cert)
                except Exception as exc:
                    failures.append(dict(**meta,error=repr(exc),traceback=traceback.format_exc()))
                write('scores.csv',rows);write('decisions.csv',decisions);write('risk_changes.csv',risks);write('relations.csv',relations);write('contrasts.csv',contrasts);save('failures.json',failures)
                save('progress.json',dict(completed=len(decisions),failed=len(failures),seconds=time.time()-start))
        for p,h in bindings.items():assert old.audit.sha(p)==h,p
        receipt.update(status='completed' if not failures else 'partial',conditions=len(decisions),failures=len(failures),CPU_hard_scoring_samples=samples,old_assets_unchanged=True)
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));save('failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        receipt.update(seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save('receipt.json',receipt)
if __name__=='__main__':main()
