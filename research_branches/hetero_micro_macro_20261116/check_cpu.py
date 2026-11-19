"""Finite verification and frozen 192-node CPU cost audit, no denoiser calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'
import hashlib
import itertools
import json
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
from objective import (make_plan,validate,descriptors,column_manifest,freeze_distribution,
                       distribution_errors,gaussian_profile)

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'results/hetero_micro_macro_cpu_20261116_v2'


def brute(A,types,plan):
    """Independent enumeration of distinct vertices; no matrix power formulas."""
    n=len(types); R=len(A)
    C=np.zeros((n,len(plan['closures'])))
    P=np.zeros((n,len(column_manifest(plan)['propagation'])))
    ci={tuple(x):i for i,x in enumerate(plan['closures'])}
    pi={tuple(x['relations'])+(x['terminal_type'],):i for i,x in enumerate(column_manifest(plan)['propagation'])}
    for i,j,k in itertools.permutations(range(n),3):
        for r in np.flatnonzero(A[:,i,j]):
            for s in np.flatnonzero(A[:,j,k]):
                P[i,pi[r,s,int(types[k])]]+=1
                for u in np.flatnonzero(A[:,k,i]):
                    if r<=u:
                        C[i,ci[r,s,u]]+=.5 if r==u else 1.
                for l in range(n):
                    if l in (i,j,k): continue
                    for u in np.flatnonzero(A[:,k,l]):
                        P[i,pi[r,s,u,int(types[l])]]+=1
    return dict(degree=A.sum(2).T,closure=C,propagation=P)


def main():
    start=time.time(); cpu=time.process_time();OUT.mkdir(exist_ok=False)
    def timeout(*_): raise TimeoutError('600 second CPU-only window')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    bindings={}
    def bind(path,expected=None):
        p=Path(path).resolve(); h=hashlib.sha256(p.read_bytes()).hexdigest()
        if expected: assert h==expected,str(p)
        bindings[str(p)]=h;return p
    def read(path):return json.loads(bind(path).read_text())
    def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    bind(__file__);bind(Path(__file__).with_name('objective.py'))
    inv=read(ROOT/'results/analytic_joint_generation_run_20261113T225800/inventory.json')
    old=read(ROOT/'results/score_tradeoff_audit_20261116_v2/asset_bindings.json')
    config=read(ROOT/'results/analytic_joint_generation_run_20261113T225800/protocol.json')
    # Exactly one previously generated graph per parent/risk-EMA model, selected
    # by smallest numeric sampling seed, not by structure quality.
    chosen=[]
    for model in sorted({x['model'] for x in inv}):
        for arm in ('parent','risk'):
            options=[x for x in inv if x['model']==model and x['arm']==arm and (arm=='parent' or x['readout']=='ema')]
            chosen.append(min(options,key=lambda x:x['sample_seed']))
    save('protocol.json',dict(scope='CPU reference objective development; no model/optimizer/sampler',
        synthetic_nodes=4,synthetic_types=[0,0,1,1],synthetic_schema=[[0,1],[0,1],[0,0],[1,1]],
        states=324, reference_graphs=2,existing_graphs=chosen,frequency_seed=2026111601,frequencies=32,
        objective_semantics='Exact expected local counts; nonlinear anonymous distribution loss is soft plug-in, not expected hard risk.',
        closure='Anchor node and ordered relation triple modulo reversal, not one global triangle number.',
        propagation='Relation-ordered simple paths of lengths 2 and 3 with terminal type; counts, not row-normalized probabilities.',
        scale='Per-type reference-only RMS, floor1 count; Fourier directions divided by sqrt(block dimensions). No output-fitted scale.',
        proposed_training='Original A CE plus separately reported macro targets; relative strength and noise-panel calibration NOT frozen or authorized.',
        limits=['Only lengths<=3, not all 5/6 rings or connectivity guarantee.','No latent subtype labels or roles.',
                'Anonymous descriptors are not complete graph invariants.','Synthetic categorical support test does not extend production relation support.',
                '192-node dense CPU reference is not a validated full-graph implementation.'],
        wall_cap_seconds=600,threads=1,RSS_cap_GiB=2,GPU_hours=0))
    types=np.array([0,0,1,1]);plan=make_plan([[0,1],[0,1],[0,0],[1,1]],2)
    slots=[];states=[];param=[]
    for i in range(4):
        for j in range(i+1,4):
            rel=[r for r,(a,b) in enumerate(plan['schema']) if sorted((a,b))==sorted((types[i],types[j]))]
            slots.append((i,j,rel));states.append([-1]+rel)
            param.extend([.19+.07*((i+j+r)%3) for r in rel])
    v=torch.tensor(param,dtype=torch.float64,requires_grad=True)
    def matrix(vv):
        P=vv.new_zeros((4,4,4));k=0
        for i,j,rs in slots:
            for r in rs:P[r,i,j]=P[r,j,i]=vv[k];k+=1
        return P
    P=matrix(v);soft=descriptors(P,types,plan)
    ref=np.zeros((4,4,4))
    for i,j,r in [(0,1,2),(1,2,0),(2,0,1),(2,3,3)]:ref[r,i,j]=ref[r,j,i]=1
    frozen=freeze_distribution({k:torch.tensor(x,dtype=torch.float64) for k,x in brute(ref,types,plan).items()},types)
    accumulated={k:np.zeros_like(x.detach().numpy()) for k,x in soft.items()}
    hard_risk=v.sum()*0;prob_sum=0.;maxerr=0.;count=0
    for assignment in itertools.product(*states):
        A=np.zeros((4,4,4));mass=v.sum()*0+1;k=0
        for (i,j,rs),r in zip(slots,assignment):
            pv=v[k:k+len(rs)];k+=len(rs)
            mass=mass*(1-pv.sum() if r<0 else pv[rs.index(r)])
            if r>=0:A[r,i,j]=A[r,j,i]=1
        expected=brute(A,types,plan)
        computed=descriptors(torch.tensor(A),types,plan)
        for name,x in expected.items():
            err=float(np.max(abs(x-computed[name].numpy()),initial=0));maxerr=max(maxerr,err)
            assert err<1e-11
            accumulated[name]+=float(mass.detach())*x
        score=torch.stack([x['error'] for x in distribution_errors(computed,types,frozen)]).mean()
        hard_risk=hard_risk+mass*score
        prob_sum+=float(mass.detach());count+=1
    assert count==324 and abs(prob_sum-1)<1e-12
    for name,x in soft.items():np.testing.assert_allclose(x.detach(),accumulated[name],atol=1e-11,rtol=1e-11)
    plugin=torch.stack([x['error'] for x in distribution_errors(soft,types,frozen)]).mean()
    g_plugin=torch.autograd.grad(plugin,v,retain_graph=True)[0]
    g_hard=torch.autograd.grad(hard_risk,v)[0]
    eps=1e-6;diff=0.
    for k in range(len(v)):
        vv=v.detach().clone();vv[k]+=eps
        plus=torch.stack([x['error'] for x in distribution_errors(descriptors(matrix(vv),types,plan),types,frozen)]).mean()
        vv[k]-=2*eps
        minus=torch.stack([x['error'] for x in distribution_errors(descriptors(matrix(vv),types,plan),types,frozen)]).mean()
        diff=max(diff,abs(float((plus-minus)/(2*eps)-g_plugin[k])))
    assert diff<1e-7
    perm=np.array([1,0,3,2])
    perm_desc=descriptors(P.detach()[:,perm][:,:,perm],types[perm],plan)
    for name in soft:np.testing.assert_allclose(perm_desc[name],soft[name].detach()[perm],atol=1e-11)
    perm_loss=torch.stack([x['error'] for x in distribution_errors(perm_desc,types[perm],frozen)]).mean()
    assert abs(float(perm_loss-plugin))<1e-12
    failures={}
    for name in ('categorical','schema','fixed_zero'):
        bad=P.detach().clone()
        try:
            if name=='categorical':bad[0,0,2]=bad[0,2,0]=.8;bad[1,0,2]=bad[1,2,0]=.8
            elif name=='schema':bad[0,0,1]=bad[0,1,0]=.1
            validate(bad,types,plan,fixed_zero=[True,False,False,False] if name=='fixed_zero' else None)
        except AssertionError:failures[name]=True
        else:raise AssertionError(name+' should reject')
    ee=torch.tensor([0.,1e-20,1e-5,.1,1.],dtype=torch.float64,requires_grad=True)
    gg=torch.autograd.grad(gaussian_profile(ee).sum(),ee)[0]
    assert torch.isfinite(gg).all()
    checks=dict(enumerated_graphs=count,mass_sum=prob_sum,hard_feature_max_error=maxerr,
        expected_feature_identity=True,plugin_finite_difference_max_error=diff,
        permutation=True,rejections=failures,gaussian_zero_limit_finite=True,
        plugin_loss=float(plugin.detach()),expected_hard_loss=float(hard_risk.detach()),
        gradient_cosine=float(torch.dot(g_plugin,g_hard)/(g_plugin.norm()*g_hard.norm())),
        plugin_is_not_expected_hard_risk=True)
    save('synthetic_checks.json',checks)
    contexts={};cost=[];rows=[]
    def load(x):
        path=Path(x);z=np.load(bind(path,old[str(path)]));ty=z['types'];schema=z['schema'];e=z['edges']
        e=e if e.shape[0]==2 else e.T
        # Frozen production assets encode edge classes 1..R; class 0 is no edge.
        # The reference operator uses relation-array indices 0..R-1.
        if 'relations' in z:
            assert np.all((z['relations']>=1)&(z['relations']<=len(schema)))
            rel=z['relations'].astype(int)-1
        else:
            # Only the already audited unique-endpoint-pair schema permits this.
            assert len({tuple(sorted(ab)) for ab in schema})==len(schema)
            rel=np.array([next(r for r,ab in enumerate(schema) if sorted(ab)==sorted(ty[[i,j]])) for i,j in e.T])
        assert len(rel)==e.shape[1]
        P=torch.zeros((len(schema),len(ty),len(ty)),dtype=torch.float64)
        for (i,j),r in zip(e.T,rel):P[int(r),i,j]=P[int(r),j,i]=1
        return P,ty,schema
    for ds in ('IMDB','PubMed'):
        parent=next(x for x in config['parents'] if x['dataset']==ds)
        P,ty,schema=load(parent['data']);pl=make_plan(schema,len(np.unique(ty)))
        tick=time.perf_counter();desc=descriptors(P,ty,pl);seconds=time.perf_counter()-tick
        froz=freeze_distribution(desc,ty)
        assert max(float(x['error']) for x in distribution_errors(desc,ty,froz))<1e-20
        edgecount=P.sum()/2
        tricount=desc['closure'].sum()/3
        assert abs(float(tricount)-float(torch.trace(P.sum(0)@P.sum(0)@P.sum(0))/6))<1e-9
        cost.append(dict(dataset=ds,nodes=len(ty),edges=float(edgecount),triangles=float(tricount),
            feature_seconds=seconds,**{k+'_dimensions':x.shape[1] for k,x in desc.items()}))
        contexts[ds]=(pl,froz,ty,schema)
        save(ds+'_columns.json',column_manifest(pl))
        save(ds+'_frozen.json',{str(t):{b:dict(scale=x['scale'].tolist(),omega=x['omega'].tolist(),
            target_real=x['target'].real.tolist(),target_imag=x['target'].imag.tolist()) for b,x in blocks.items()} for t,blocks in froz.items()})
    for item in chosen:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024*1024:raise MemoryError('2GiB cap')
        P,ty,schema=load(item['graph']);pl,froz,ref_ty,ref_schema=contexts[item['dataset']]
        np.testing.assert_array_equal(ty,ref_ty);np.testing.assert_array_equal(schema,ref_schema)
        tick=time.perf_counter();desc=descriptors(P,ty,pl);seconds=time.perf_counter()-tick
        for x in distribution_errors(desc,ty,froz):
            rows.append(dict(**{k:item[k] for k in ('model','dataset','train_seed','arm','readout','sample_seed')},
                type=x['type'],block=x['block'],error=float(x['error']),feature_seconds=seconds))
    save('real192_reference_cost.json',cost);save('existing_graph_scores.json',rows)
    for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('asset_bindings.json',bindings)
    save('completion.json',dict(status='complete',wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU_hours=0,training_updates=0,
        synthetic_enumerated_graphs=324,reference_graphs=2,existing_generated_graphs=8,new_generated_graphs=0,
        autograd='CPU descriptor/analytic objective only; no neural network calls',original_assets_unchanged=True))
    print(json.dumps(checks));print(json.dumps(cost));signal.alarm(0)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc))))
        raise
