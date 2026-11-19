"""Summarize retained CPU samples and independently verify saved LOO vectors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/log_risk_cpu_20261111/run2'

def main():
    start=time.time();protocol=json.loads((OUT/'protocol.json').read_text())
    rows=json.loads((OUT/'panel_summary.json').read_text());graphs=json.loads((OUT/'existing_graph_scores.json').read_text())
    maxerror=0.;checks=0
    for index,item in enumerate(protocol['panel']):
        z=np.load(OUT/f'panel_{index:02d}.npz');p=np.load(item['path'])['probability']
        for rep in [0,15]:
            key=0 if item['kind']=='same_training_input_u2012' else 1
            rng=np.random.default_rng(np.random.SeedSequence([2026111151,item['seed'],key,item['index'],rep]))
            bits=(rng.random((16,len(p)))<p).astype(float)
            for arm in ['old','new']:
                cost=z[arm+'_costs'][16*rep:16*(rep+1)].mean(1)
                gradient=(cost-cost.mean()) @ (bits-bits.mean(0))/15
                err=float(abs(gradient-z[arm+'_logit_gradients'][rep]).max())
                np.testing.assert_allclose(gradient,z[arm+'_logit_gradients'][rep],atol=1e-9,rtol=1e-10)
                maxerror=max(maxerror,err);checks+=1
    paired=[]
    for task in sorted({r['task'] for r in rows}):
        selected=[r for r in rows if r['task']==task and r['kind']=='same_training_input_u2012']
        record=dict(task=task)
        for phase in ['before','after']:
            group=[r for r in selected if r['phase']==phase]
            for metric in ['old_score','new_score','expected_edges','old_dominant_share','new_dominant_share']:
                record[phase+'_'+metric]=float(np.mean([r[metric] for r in group]))
        paired.append(record)
    gsummary=[]
    for task in sorted({r['task'] for r in graphs}):
        for arm in sorted({r['arm'] for r in graphs}):
            group=[r for r in graphs if r['task']==task and r['arm']==arm]
            if not group:continue
            gsummary.append(dict(task=task,arm=arm,n=len(group),
                old_mean=float(np.mean([r['old_score'] for r in group])),
                new_mean=float(np.mean([r['new_score'] for r in group])),
                edges_mean=float(np.mean([r['edges'] for r in group]))))
    reversals=[]
    for task in sorted({r['task'] for r in graphs}):
        g=[r for r in graphs if r['task']==task and r['path']]
        for i,a in enumerate(g):
            for b in g[i+1:]:
                if (a['old_score']-b['old_score'])*(a['new_score']-b['new_score']) < -1e-12:
                    reversals.append(dict(task=task,first=a['path'],second=b['path'],first_old=a['old_score'],second_old=b['old_score'],first_new=a['new_score'],second_new=b['new_score']))
    reliability={}
    for arm in ['old','new']:
        active=[r for r in rows if r[arm+'_split_half_cosine'] is not None]
        reliability[arm]=dict(nonzero_panels=len(active),
            split_half_cosine_median=float(np.median([r[arm+'_split_half_cosine'] for r in active])),
            split_half_cosine_min=float(min(r[arm+'_split_half_cosine'] for r in active)),
            split_half_cosine_max=float(max(r[arm+'_split_half_cosine'] for r in active)),
            negative_split_half=int(sum(r[arm+'_split_half_cosine']<0 for r in active)),
            mean_SE_over_norm_median=float(np.median([r[arm+'_mean_SE_over_norm'] for r in active])),
            zero_repeats=int(sum(r[arm+'_zero_gradient_repeats'] for r in rows)))
    result=dict(same_input_pairs=paired,existing_graph_means=gsummary,rank_reversals=reversals,
        logit_reliability=reliability,independent_covariance_checks=checks,max_abs_error=maxerror,
        seconds=time.time()-start,decision='formula and scale checks pass; all-MC feedback not yet reliable enough to claim readiness; no GPU/model update tested')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
