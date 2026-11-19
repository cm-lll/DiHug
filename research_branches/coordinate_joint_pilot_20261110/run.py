"""CPU local-coordinate pilot. No neural calls or optimizer updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,csv,itertools,resource,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
import importlib.util
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
audit=mod('probability_audit',ROOT/'research_branches/density_arrangement_audit_20261110/run.py')
exactmod=mod('analytic',ROOT/'research_branches/exact_hard_moments_20261110/moments.py');torch=exactmod.torch
remaining=mod('remaining',ROOT/'research_branches/exact_hard_moments_20261110/remaining.py').remaining_costs
loo=mod('loo',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py').loo_gradient
OUT=ROOT/'results/coordinate_joint_pilot_20261110';PREV=ROOT/'results/displacement_response_20261110'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def write(n,rows):
    if rows:
        with (OUT/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def project(C,W):
    live=W>0;s=np.ones(2);n=int(live.sum())
    if not n:return np.zeros(2),dict(status='zero_direction',feasibility=0.)
    root=np.sqrt(W[live]);a=C[:,live]/root;norm=np.linalg.norm(a,axis=1);a=a[norm>0]/norm[norm>0,None]
    A=np.vstack([a,np.eye(n),-np.eye(n)]);b=np.r_[np.zeros(len(a)),root,np.zeros(n)];tol=1e-10*max(1,float(root.max()));choices=[np.zeros(n)]
    if np.max(A@root-b)<=tol:choices.append(root.copy())
    for count in range(1,n+1):
        for ix in itertools.combinations(range(len(b)),count):
            M=A[list(ix)];rhs=M@root-b[list(ix)];H=M@M.T
            if np.linalg.matrix_rank(H,tol=1e-12)<count:continue
            lam=np.linalg.solve(H,rhs);y=root-M.T@lam
            if np.max(abs(M@y-b[list(ix)]))<=tol and np.max(A@y-b)<=tol:choices.append(y)
    y=min(choices,key=lambda y:float(np.sum((y-root)**2)));s[live]=np.clip(y/root,0,1);s[~live]=0
    violation=float(np.max(A@(s[live]*root)-b));assert violation<=tol*2
    return s,dict(status='solved',feasibility=violation,tolerance=tol,distance_squared=float(np.sum(((s[live]-1)*root)**2)))
def decompose(p0,p1,f):
    assert np.array_equal(p0==0,p1==0) and np.array_equal(p0==1,p1==1)
    active=(p0>0)&(p0<1);z=np.zeros(len(p0));d=np.zeros(len(p0));z[active]=np.log(p0[active])-np.log1p(-p0[active]);d[active]=np.log(p1[active])-np.log1p(-p1[active])-z[active]
    w=p0*(1-p0);density=np.zeros_like(d)
    for family in np.unique(f):
        ix=(f==family)&active;v=w[ix].sum()
        if v>0:density[ix]=(w[ix]*d[ix]).sum()/v
    arr=d-density;J=np.stack([density,arr]);W=(J*J*w).sum(1)
    assert abs((w*density*arr).sum())<1e-10*max(1,float(W.sum()))
    return active,z,d,J,W
def probability(p0,p1,active,z,J,s):
    if np.array_equal(s,np.ones(2)):return p1.copy()
    if not s.any():return p0.copy()
    p=p0.copy();p[active]=expit(z[active]+(s@J)[active]);assert ((p[active]>0)&(p[active]<1)).all();return p
def tests():
    rng=np.random.default_rng(2026111073);errors=[]
    for i in range(30):
        C=rng.normal(size=(5,2));W=np.exp(rng.normal(size=2));s,cert=project(C,W)
        res=minimize(lambda x:.5*np.sum(W*(x-1)**2),np.zeros(2),jac=lambda x:W*(x-1),bounds=[(0,1)]*2,constraints=[dict(type='ineq',fun=lambda x:-C@x,jac=lambda x:-C)],method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
        assert res.success;err=abs(.5*np.sum(W*(s-1)**2)-res.fun);assert err<1e-8;errors.append(err)
    p0=np.array([0,.1,.3,.6,.7,1.]);p1=np.array([0,.2,.25,.65,.4,1.]);f=np.array([0,0,0,1,1,1]);a,z,d,J,W=decompose(p0,p1,f);eps=1e-6;q=probability(p0,p1,a,z,J,np.array([0,eps]));slope=max(abs((q-p0)[f==k].sum()/eps) for k in (0,1));assert slope<1e-6
    np.testing.assert_allclose(probability(p0,p1,a,z,J,np.ones(2)),p1,atol=1e-15)
    return dict(passed=True,independent_solver_cases=30,max_objective_difference=max(errors),arrangement_count_derivative_error=slope,endpoint_support_and_reconstruction=True)
def main():
    start=time.time();OUT.mkdir(exist_ok=False);receipt=dict(status='running',GPU_hours=0,model_calls=0,network_backwards=0,optimizer_updates=0,reverse_chains=0);rows=[];states=[];risks=[];relations=[];failures=[];samples=0
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        ap=json.loads((ROOT/'results/density_arrangement_audit_20261110/protocol.json').read_text());records=ap['conditions_manifest'];proto=json.loads((PREV/'protocol.json').read_text());bindings=json.loads((ROOT/'results/density_arrangement_audit_20261110/bindings.json').read_text())
        for p,h in bindings.items():assert audit.sha(p)==h,p
        for p in [Path(__file__),ROOT/'research_branches/exact_hard_moments_20261110/moments.py',ROOT/'research_branches/exact_hard_moments_20261110/remaining.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py']:bindings[str(p)]=audit.sha(p)
        for row in proto['tasks']:
            name=f"{row['dataset']}_{row['seed']}";source=Path(row['response_source'])
            for label in ap['panels']:
                for p in [source/f'{label}_before_scored.npz',source/f'{label}_after_scored.npz']+[PREV/name/f'a{a}_{label}_scored.npz' for a in (.25,.5)]:bindings[str(p)]=audit.sha(p)
        save('protocol.json',dict(start_epoch=start,CPU_science_limit_seconds=540,process_limit_seconds=600,conditions=84,objectives='unchanged five risks, three exact plus two LOO estimates',rule='Fisher-orthogonal relation common logit change and within-relation residual; nearest retention s in [0,1]^2 satisfying five first-order constraints',direction_draws=128,evaluation_draws=128,split='first/last halves of old frozen 256 uniforms, disjoint draws; old development inputs, not independent confirmation',controls=['unchanged','original','coordinate_projected','same_Fisher_logit_norm_uniform_scale'],no_line_search=True,no_finite_risk_acceptance=True,no_tuning=True,network_calls=0,GPU_hours=0));save('bindings.json',bindings);save('cpu_gate.json',tests())
        for row in proto['tasks']:
            ds,seed=row['dataset'],row['seed'];name=f'{ds}_{seed}';source=Path(row['response_source']);subset=[r for r in records if r['dataset']==ds and r['seed']==seed];data=np.load(row['data']);first=np.load(subset[0]['before']);core=audit.HardScore(data['types'],first['query'],first['families'],first['labels'],len(data['schema']));moment=audit.Moments(core);exact=exactmod.ExactHardMoments(core)
            for label in ap['panels']:
                sub=[r for r in subset if r['panel']==label];p0=np.load(sub[0]['before'])['probability'];bc=np.load(source/f'{label}_before_scored.npz');baseV=bc['values'];baseC=bc['costs_remaining'];j=ap['panels'].index(label)-1;es=seed+(proto['same_input_evaluation_seed_offset'] if label=='same' else proto['evaluation_hard_seed_offset']+j);u=np.random.default_rng(es).random((256,len(p0)))
                grad=np.zeros((5,len(p0)));v,G,_=exact.value_gradient(p0);grad[[0,2,3]]=G;grad[[1,4]]=loo(p0,(u[:128]<p0).astype(np.uint8),baseC[:128]);np.testing.assert_allclose(v,baseV[[0,2,3]],atol=1e-11,rtol=1e-9)
                before=baseV.copy();before[[1,4]]=baseC[128:].mean(0)
                for rec in sub:
                    if time.time()-start>540:raise TimeoutError('CPU science stop')
                    p1=np.load(rec['after'])['probability'];meta={k:rec[k] for k in ('dataset','seed','panel','t','alpha')}
                    if not np.array_equal(p0==0,p1==0) or not np.array_equal(p0==1,p1==1):failures.append(dict(**meta,error='support changed'));continue
                    active,z,d,J,W=decompose(p0,p1,core.f);C=grad@J.T;s,cert=project(C,W);beta=float(np.sqrt(np.sum(W*s*s)/W.sum())) if W.sum() else 0.;pc=probability(p0,p1,active,z,J,s);pm=probability(p0,p1,active,z,J,np.array([beta,beta]));inner=C@s
                    costpath=source/f'{label}_after_scored.npz' if rec['alpha']==1 else PREV/name/f'a{rec["alpha"]}_{label}_scored.npz';ac=np.load(costpath);after=ac['values'].copy();afterC=ac['costs_remaining'][128:];after[[1,4]]=afterC.mean(0)
                    vals={'unchanged':(before,baseC[128:]),'original':(after,afterC)};probs={'unchanged':p0,'original':p1,'coordinate_projected':pc,'same_norm_scale':pm}
                    for arm,p in [('coordinate_projected',pc),('same_norm_scale',pm)]:
                        reuse=next((name for name,q in probs.items() if name in vals and np.array_equal(q,p)),None)
                        if reuse:vals[arm]=vals[reuse];continue
                        V,_,_=moment.score(p);cost=remaining(core,(u[128:]<p).astype(np.uint8));samples+=128;value=np.zeros(5);value[[0,2,3]]=V;value[[1,4]]=cost.mean(0);vals[arm]=(value,cost)
                    states.append(dict(**meta,s_density=float(s[0]),s_arrangement=float(s[1]),same_norm_beta=beta,original_Fisher_norm=float(np.sqrt(W.sum())),retained_Fisher_norm=float(np.sqrt((W*s*s).sum())),zero=bool(not s.any()),unchanged_direction=bool(np.array_equal(s,np.ones(2))),direction_changed=bool(abs(s[0]-s[1])>1e-8),first_order_max=float(inner.max()),solver_residual=cert['feasibility']))
                    for arm,(value,cost) in vals.items():
                        rows.append(dict(**meta,arm=arm,**{n:float(x) for n,x in zip(core.names,value)},mean_five=float(value.mean()),expected_edges=float(probs[arm].sum())))
                        for k,n in enumerate(core.names):risks.append(dict(**meta,arm=arm,metric=n,before=float(before[k]),value=float(value[k]),change=float(value[k]-before[k]),analytic=k in (0,2,3),paired_MC_SE=float((cost-baseC[128:])[:,[1,4].index(k)].std(ddof=1)/np.sqrt(128)) if k in (1,4) else 0.))
                        for f in range(core.r):
                            mask=core.f==f;pos=mask&(core.reference==1);neg=mask&(core.reference==0);p=probs[arm];relations.append(dict(**meta,arm=arm,family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),expected_edges=float(p[mask].sum()),positive_missed_mass=float((1-p[pos]).sum()),negative_false_mass=float(p[neg].sum())))
                    np.savez_compressed(OUT/f'{ds}_{seed}_{label}_a{rec["alpha"]}.npz',C=C,W=W,s=s,coefficients_first_order=inner,probability_projected=pc,probability_matched=pm,values=np.stack([vals[k][0] for k in vals]),arms=np.array(list(vals)))
                    save('progress.json',dict(completed=len(states),seconds=time.time()-start,CPU_hard_scoring_samples=samples))
        for p,h in bindings.items():assert audit.sha(p)==h,p
        write('scores.csv',rows);write('decisions.csv',states);write('risk_changes.csv',risks);write('relations.csv',relations);save('failures.json',failures)
        receipt.update(status='completed',conditions=len(states),skipped=len(failures),CPU_hard_scoring_samples=samples,old_assets_unchanged=True)
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));(OUT/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);save('receipt.json',receipt)
if __name__=='__main__':main()
