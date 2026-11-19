"""Same log1p objective; exact existing risk moments as control variates.

No fitted parameters, neural forward/backward, optimizer or reverse sampling.
All hard draws/costs are reused from the frozen primary CPU audit.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import importlib.util,json,time,resource,hashlib
from pathlib import Path
import numpy as np
import torch
import run as base
torch.set_num_threads(1)
ROOT=base.ROOT;OUT=base.OUT
spec=importlib.util.spec_from_file_location('frozen_exact_moments',ROOT/'research_branches/exact_hard_moments_20261110/moments.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
J=np.array([0,2,3])


def reliability(gg):
    g=gg.mean(0);norm=float(np.linalg.norm(g));var=float(((gg-g)**2).sum()/15)
    a=gg[:8].mean(0);b=gg[8:].mean(0);den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return dict(mean_gradient_norm=norm,variance_trace=var,
        mean_SE_over_norm=float(np.sqrt(var/16)/norm) if norm>1e-30 else None,
        split_half_cosine=float(a@b/den) if den>1e-30 else None,
        zero_repeats=int((np.linalg.norm(gg,axis=1)==0).sum()))


def main():
    started=time.time();dest=OUT/'control_variate';dest.mkdir(exist_ok=False)
    proto=json.loads((OUT/'protocol.json').read_text());tasks=json.loads((base.OLD/'protocol.json').read_text())['tasks']
    definition=dict(objective_unchanged=True,components_with_exact_moments=J.tolist(),
        a='a_j=1/(1+mu_j), mu_j=exact E[C_j], detached; no searched coefficient',
        gradient='LOO(mean(log1p(C))-sum_j(a_j*C_j)/5) + sum_j(a_j*grad E[C_j])/5',
        input_protocol_sha256=base.sha(OUT/'protocol.json'),code_sha256=base.sha(__file__),
        exact_code_sha256=base.sha(module.__file__),hard_draws='same 16x16 retained draws, no extra sampling',
        resources='single CPU thread, 10min internal bound, no CUDA/model/optimizer')
    (dest/'protocol.json').write_text(json.dumps(definition,indent=2))
    # Exact pair expectation on 64 states verifies the estimator, including
    # detached coefficients that are functions of p. No model is involved.
    import itertools
    ty=np.array([0,0,1,1]);q=np.array(list(itertools.combinations(range(4),2))).T
    f=np.array([ty[u]+ty[v] for u,v in q.T]);c=base.FastScore(ty,q,f,np.array([1,1,0,1,0,1]),3)
    bits=np.array(list(itertools.product([0,1],repeat=6)),dtype=float);cost=c.batch(bits)
    p=np.linspace(.1,.8,6);prob=np.prod(np.where(bits,p,1-p),1);mu=prob@cost
    exact=(prob[:,None]*cost).T@(bits-p);truth=(prob[:,None]*np.log1p(cost)).T@(bits-p)
    a=1/(1+mu[J]);residual=np.log1p(cost).mean(1)-cost[:,J]@a/5
    pair=np.zeros(6)
    for i in range(64):pair+=prob[i]*((prob*(residual[i]-residual))@(bits[i]-bits))/2
    pair+=a@exact[J]/5
    np.testing.assert_allclose(pair,truth.mean(0),rtol=1e-11,atol=1e-13)
    (dest/'formula_check.json').write_text(json.dumps(dict(exhaustive_LOO_CV_max_error=float(abs(pair-truth.mean(0)).max()),passed=True)))
    cores={};exacts={};rows=[]
    for itemindex,item in enumerate(proto['panel']):
        if time.time()-started>600:raise TimeoutError('CV CPU limit')
        tid=item['task'];task=next(t for t in tasks if t['id']==tid);ds=item['dataset']
        if ds not in cores:
            z=np.load(base.OLD/tid/'validation_u2000_raw_t1.npz');data=np.load(task['parent']['data'])
            cores[ds]=base.FastScore(data['types'],z['query'],z['families'],z['labels'],len(data['schema']))
            exacts[ds]=module.ExactHardMoments(cores[ds])
        p=np.load(item['path'])['probability'];z=np.load(OUT/f'panel_{itemindex:02d}.npz')
        mu,eg,_=exacts[ds].value_gradient(p);a=1/(1+mu);cv=[];hybrid=[]
        for rep in range(16):
            key=0 if item['kind']=='same_training_input_u2012' else 1
            rng=np.random.default_rng(np.random.SeedSequence([2026111151,item['seed'],key,item['index'],rep]))
            bits=(rng.random((16,len(p)))<p).astype(float)
            cost=z['old_costs'][16*rep:16*(rep+1)]
            residual=np.log1p(cost).mean(1)-cost[:,J]@a/5
            # Covariance form is independent of the original implementation.
            v=(residual-residual.mean())@(bits-bits.mean(0))/15+(a@eg)/5
            cv.append(v)
            residual_old=cost[:,[1,4]].sum(1)/5
            hybrid.append((residual_old-residual_old.mean())@(bits-bits.mean(0))/15+eg.sum(0)/5)
        cv=np.asarray(cv);hybrid=np.asarray(hybrid)
        row={k:v for k,v in item.items() if k!='path'}
        row.update(exact_mu=mu.tolist(),coefficients=a.tolist(),new_CV=reliability(cv),old_hybrid=reliability(hybrid),
                   new_all_MC=reliability(z['new_logit_gradients']))
        rows.append(row);np.savez_compressed(dest/f'panel_{itemindex:02d}.npz',new_CV=cv,old_hybrid=hybrid,exact_mu=mu,exact_gradient=eg,coefficients=a)
        (dest/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        print(json.dumps(dict(panel=itemindex+1,seconds=round(time.time()-started,2),new_CV_cos=row['new_CV']['split_half_cosine'])),flush=True)
    summary={}
    for name in ['new_CV','old_hybrid','new_all_MC']:
        active=[r[name] for r in rows if r[name]['split_half_cosine'] is not None]
        summary[name]=dict(nonzero_panels=len(active),median_split_half_cosine=float(np.median([r['split_half_cosine'] for r in active])),
            min_split_half_cosine=float(min(r['split_half_cosine'] for r in active)),negative_split_half=sum(r['split_half_cosine']<0 for r in active),
            median_mean_SE_over_norm=float(np.median([r['mean_SE_over_norm'] for r in active])),zero_repeats=sum(r[name]['zero_repeats'] for r in rows))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    receipt=dict(seconds=time.time()-started,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 panels=len(rows),GPU_hours=0,model_calls=0,updates=0,extra_hard_samples=0,
                 torch_cuda_initialized=torch.cuda.is_initialized(),status='completed')
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
