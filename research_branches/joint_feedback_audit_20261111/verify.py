"""Reconstruct draws and gradients independently; no rescoring or model calls."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,time,resource
import numpy as np

root=Path(__file__).resolve().parents[2]
out=root/'results/joint_feedback_audit_20261111'
protocol=json.loads((out/'protocol.json').read_text())
rows=json.loads((out/'panel_summary.json').read_text())
start=time.perf_counter();maxerr=0.;comparisons=[]
for index,item in enumerate(protocol['panel']):
    z=np.load(item['path']);saved=np.load(out/f'panel_{index:02d}.npz')
    p=z['probability'];families=z['families'];ref=z['labels'];grad=[];cg=[];counts=[]
    target=np.bincount(families,weights=ref,minlength=10)
    for rep in range(protocol['repeats']):
        rng=np.random.default_rng(np.random.SeedSequence([protocol['random_root'],item['train_seed'],item['t'],rep]))
        b=(rng.random((protocol['K'],len(p)))<p).astype(float)
        costs=saved['joint_costs'][rep*16:(rep+1)*16]
        # Sample covariance identity for leave-one-out estimator.
        grad.append((b-b.mean(0)).T@(costs-costs.mean())/15)
        c=np.stack([b[:,families==r].sum(1) for r in range(10)],axis=1)
        count_cost=np.mean((c-target)**2/np.maximum(1,target)**2,axis=1)
        cg.append((b-b.mean(0)).T@(count_cost-count_cost.mean())/15)
        counts.extend(b.sum(1))
    err=float(np.max(abs(np.asarray(grad)-saved['joint_logit_gradients'])))
    maxerr=max(maxerr,err)
    np.testing.assert_allclose(grad,saved['joint_logit_gradients'],atol=1e-14,rtol=1e-10)
    np.testing.assert_allclose(cg,saved['count_MC_logit_gradients'],atol=1e-13,rtol=1e-10)
    np.testing.assert_array_equal(counts,saved['sample_edge_counts'])
    np.testing.assert_allclose(saved['type_costs'].mean(1),saved['joint_costs'],atol=1e-14)
    # Exact count derivative by conditioning the Bernoulli slot to 1 versus 0.
    delta=np.zeros_like(p)
    for r in range(10):
        sel=families==r;mu=p[sel].sum();m=target[r]
        delta[sel]=((2*(mu-p[sel]-m)+1)/max(1,m)**2/10)*p[sel]*(1-p[sel])
    np.testing.assert_allclose(delta,saved['count_exact_logit_gradient'],atol=1e-14)
    mc=np.mean(cg,0);exact=saved['count_exact_logit_gradient']
    cos=float(mc@exact/(np.linalg.norm(mc)*np.linalg.norm(exact)))
    comparisons.append(dict(dataset=item['dataset'],train_seed=item['train_seed'],t=item['t'],
        count_MC_to_exact_cosine=cos,count_MC_relative_error=float(np.linalg.norm(mc-exact)/np.linalg.norm(exact))))
for path,h in protocol['bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
result=dict(status='pass',conditions=len(rows),reconstructed_draws=1536,gradient_max_absolute_error=maxerr,
    covariance_LOO_equivalence=True,exact_count_conditioning_check=True,source_hashes_unchanged=True,
    seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    count_MC_comparison=comparisons,limitations='Independent algebra and RNG reconstruction; not independent full heat feature implementation or real-graph exact joint gradient.')
(out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
