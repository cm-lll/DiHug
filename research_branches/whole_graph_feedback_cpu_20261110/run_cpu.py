import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import csv
import hashlib
import itertools
import json
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/whole_graph_feedback_cpu_20261110'
SOURCE = ROOT / 'research_branches/hard_structure_rebar_cpu_20261108/objective.py'
sys.path.insert(0, str(SOURCE.parent))
from objective import JointLocalScore, estimate_parts
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
CFG = json.loads((HERE / 'protocol.json').read_text())
START = time.time()
COUNTS = dict(GPU_hours=0, production_model_calls=0, optimizer_updates=0,
              reverse_chains=0, auxiliary_hard_samples=0)

def dump(name, x):
    (OUT / name).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))

def table(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def guard():
    if time.time()-START > CFG['wall_seconds']-5:
        raise TimeoutError('CPU budget reached; no reduced panel masquerading as complete')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > CFG['RSS_GiB']*1024**2:
        raise MemoryError('CPU RSS cap reached')

def main():
    OUT.mkdir(exist_ok=False)
    dump('protocol.json', CFG)
    paths = [SOURCE, HERE/'run_cpu.py', HERE/'protocol.json']
    binding = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    dump('bindings.json', binding)
    types = np.array(CFG['fixture']['types'])
    q = np.array(np.triu_indices(len(types), 1))
    pairs = sorted(set(tuple(sorted((types[i],types[j]))) for i,j in q.T))
    family = np.array([pairs.index(tuple(sorted((types[i],types[j])))) for i,j in q.T])
    edges = set(map(tuple, CFG['fixture']['reference_edges']))
    y = torch.tensor([float(tuple(e) in edges) for e in q.T])
    score = JointLocalScore(types,q,family,y,len(pairs))
    m = len(y)
    bits = torch.tensor(list(itertools.product([0.,1.],repeat=m)))
    with torch.no_grad(): values = score(bits)
    # Independent counts for every state; protects reused feature implementation.
    feature = score.features(bits).numpy()
    manual = np.zeros_like(feature)
    for s,b in enumerate(bits.numpy()):
        present = {tuple(q[:,e]) for e in range(m) if b[e]}
        for e,(i,j) in enumerate(q.T):
            if b[e]: manual[s,i,family[e]]+=1; manual[s,j,family[e]]+=1
        for tri in itertools.combinations(range(len(types)),3):
            if all(e in present for e in itertools.combinations(tri,2)):
                for i in tri:
                    other = tuple(sorted(types[j] for j in tri if j!=i))
                    manual[s,i,len(pairs)+score.triangle_pairs.index(other)] += 1
    assert np.array_equal(feature,manual)
    J = np.random.default_rng(CFG['jacobian_seed']).normal(size=(m,4))
    J[:,0] = 1
    J /= np.linalg.norm(J,axis=0,keepdims=True)
    np.savez_compressed(OUT/'exact_fixture.npz', bits=bits.numpy(), costs=values.numpy(),
                        query=q, families=family, reference=y.numpy(), jacobian=J)
    rows=[]; checks=[]; timing=[]; profiles=[]
    plist=[torch.full((m,),.03),torch.full((m,),.30),torch.where(y.bool(),.95,.01)]
    for pi,(name,p) in enumerate(zip(CFG['profiles'],plist)):
        guard()
        z=torch.logit(p).requires_grad_(True)
        mass=torch.where(bits.bool(),z.sigmoid(),1-z.sigmoid()).prod(-1)
        risk=(mass*values).sum()
        exact=torch.autograd.grad(risk,z)[0].detach().numpy()
        rf=values[:,None]*(bits-p)
        enumeration=(mass.detach()[:,None]*rf).sum(0).numpy()
        softz=z.detach().clone().requires_grad_(True)
        softgrad=torch.autograd.grad(score(softz.sigmoid()),softz)[0].detach().numpy()
        cond=[]
        ids=np.arange(len(bits)); weights=2**np.arange(m-1,-1,-1)
        for e in range(m):
            lo=ids & ~weights[e]; hi=ids | weights[e]
            cond.append((p[e]*(1-p[e])*(values[hi]-values[lo])).numpy())
        cond=np.stack(cond,-1)
        cm=(mass.detach().numpy()[:,None]*cond).sum(0)
        fd=[]
        def exact_risk(zz):
            pp=torch.tensor(zz).sigmoid()
            return float((torch.where(bits.bool(),pp,1-pp).prod(-1)*values).sum())
        for e in range(m):
            v=np.zeros(m);v[e]=1e-5
            fd.append((exact_risk(z.detach().numpy()+v)-exact_risk(z.detach().numpy()-v))/2e-5)
        err=max(np.max(abs(exact-enumeration)),np.max(abs(exact-cm)),np.max(abs(exact-np.array(fd))))
        assert err < 1e-8
        assert abs(float(mass.sum())-1)<1e-12
        checks.append(dict(profile=name,max_gradient_error=float(err),feature_count_error=0))
        devgen=torch.Generator().manual_seed(CFG['development_seed']+pi)
        def parts(n,gen):
            COUNTS['auxiliary_hard_samples']+=n
            u=torch.rand((n,m),generator=gen);v=torch.rand((n,m),generator=gen)
            return estimate_parts(score,p.expand(n,-1),u,v,CFG['temperature'])
        dev={k:[] for k in ['rf_base','control']}
        for _ in range(CFG['development_samples']//256):
            guard();r=parts(256,devgen)
            for k in dev:dev[k].append(r[k].numpy())
        d={k:np.concatenate(v) for k,v in dev.items()}
        a=d['rf_base']-d['rf_base'].mean(0); c=d['control']-d['control'].mean(0)
        eta=-float((a*c).sum())/float((c*c).sum()) if (c*c).sum()>0 else 0.
        # Freeze eta before reading any evaluation draws.
        profile=dict(profile=name,probability=p.tolist(),eta=eta,exact_hard_risk=float(risk.detach()),
                     exact_gradient=exact.tolist(),soft_gradient=softgrad.tolist(),
                     soft_gradient_bias_norm=float(np.linalg.norm(softgrad-exact)),
                     conditional_single_draw_trace_variance=float((mass.detach().numpy()[:,None]*(cond-exact)**2).sum()))
        profiles.append(profile); dump('profiles.json',profiles)
        for K in CFG['batch_sizes']:
            guard()
            gen=torch.Generator().manual_seed(CFG['evaluation_seed']+pi*1000+K)
            total=K*CFG['evaluation_batches']; chunks=[]
            for start in range(0,total,256):
                guard();chunks.append(parts(min(256,total-start),gen))
            data={k:torch.cat([r[k] for r in chunks]).numpy().reshape(CFG['evaluation_batches'],K,-1)
                  for k in ['rf','rf_base','control','bits','hard_value']}
            cv=data['hard_value']
            loo=(cv.sum(1,keepdims=True)-cv)/(K-1)
            estimators={
                'raw_score_function':data['rf'].mean(1),
                'detached_soft_baseline':data['rf_base'].mean(1),
                'leave_one_out':((cv-loo)*(data['bits']-p.numpy())).mean(1),
                'REBAR_development_coefficient':(data['rf_base']+eta*data['control']).mean(1),
                'deterministic_soft_gradient_as_biased_reference':np.broadcast_to(softgrad,(CFG['evaluation_batches'],m))}
            archive={**estimators,'exact':exact,'sampled_bits':data['bits'],'costs':cv}
            np.savez_compressed(OUT/f'{name}_K{K}.npz',**archive)
            for method,est in estimators.items():
                for space,transform in [('logit',np.eye(m)),('synthetic_shared_parameters',J)]:
                    target=exact@transform; estimate=est@transform
                    dot=estimate@target; norm=np.linalg.norm(estimate,axis=1)
                    cosine=dot/np.maximum(norm*np.linalg.norm(target),1e-30)
                    variance=estimate.var(0,ddof=1)
                    mean=estimate.mean(0);se=np.sqrt(variance/len(estimate))
                    rows.append(dict(profile=name,K=K,method=method,space=space,
                        MSE=float(((estimate-target)**2).sum(1).mean()),
                        trace_variance=float(variance.sum()),mean_bias_norm=float(np.linalg.norm(mean-target)),
                        max_mean_error_SE=float(np.max(abs(mean-target)/np.maximum(se,1e-15))),
                        mean_cosine=float(cosine.mean()),descent_fraction=float((dot>1e-16).mean()),
                        zero_estimate_fraction=float((norm<1e-14).mean()),
                        mean_coordinates_missing_rare_outcome=float(np.where(p.numpy()<.5,
                            data['bits'].sum(1)==0,data['bits'].sum(1)==K).sum(1).mean())))
        # Standalone timings: no cached hard score or enumeration used in methods.
        tg=torch.Generator().manual_seed(CFG['evaluation_seed']+9000+pi)
        base=score(p).detach()
        for repeat in range(14):
            for method in (['leave_one_out','detached_soft_baseline','REBAR_development_coefficient']
                           if repeat%2==0 else ['REBAR_development_coefficient','detached_soft_baseline','leave_one_out']):
                tick=time.perf_counter()
                if method=='REBAR_development_coefficient':
                    r=parts(16,tg);ans=(r['rf_base']+eta*r['control']).mean(0)
                else:
                    b=(torch.rand((16,m),generator=tg)<p).to(p.dtype)
                    COUNTS['auxiliary_hard_samples']+=16
                    with torch.no_grad():cost=score(b)
                    baseline=(cost.sum()-cost)/15 if method=='leave_one_out' else base
                    ans=((cost-baseline)[:,None]*(b-p)).mean(0)
                elapsed=time.perf_counter()-tick
                assert torch.isfinite(ans).all()
                if repeat>=2:timing.append(dict(profile=name,method=method,repeat=repeat-2,
                                               K=16,seconds=elapsed))
        table('results.csv',rows);table('timing.csv',timing);dump('checks.json',checks)
    costs=[]
    for name in CFG['profiles']:
        tm={method:float(np.median([r['seconds'] for r in timing if r['profile']==name and r['method']==method]))
            for method in ['leave_one_out','detached_soft_baseline','REBAR_development_coefficient']}
        for space in ['logit','synthetic_shared_parameters']:
            selected=[r for r in rows if r['profile']==name and r['K']==16 and r['space']==space and r['method'] in tm]
            ref=next(r for r in selected if r['method']=='leave_one_out')
            for r in selected:
                costs.append(dict(profile=name,space=space,method=r['method'],batch_seconds=tm[r['method']],
                    variance_time_ratio_to_LOO=r['trace_variance']*tm[r['method']]/(ref['trace_variance']*tm['leave_one_out']),
                    MSE_ratio_to_LOO=r['MSE']/ref['MSE']))
    table('cost_comparison.csv',costs)
    for path,h in binding.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    dump('receipt.json',dict(status='complete',wall_seconds=time.time()-START,
         peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         counts=COUNTS,torch_version=torch.__version__,CUDA_initialized=torch.cuda.is_initialized(),
         max_exact_check_error=max(r['max_gradient_error'] for r in checks),
         limits='180 CPU-worker seconds / 2 GiB RSS monitored per chunk; no GPU'))
    print(json.dumps(json.loads((OUT/'receipt.json').read_text()),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.time()-START,counts=COUNTS))
        raise
