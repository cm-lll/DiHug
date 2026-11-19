import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,csv,itertools,resource
from pathlib import Path
import numpy as np
import torch
from run import ROOT,OUT,CPU,HardScore,ExactHardMoments,module,sha,write,dump
from remaining import remaining_costs
loo=module('stable_loo',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py').loo_gradient
def records(n):return list(csv.DictReader((OUT/n).open()))
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert json.loads((OUT/'receipt.json').read_text())['status']=='completed'
    # Additional support-boundary implementation tests, no change to frozen real panel.
    edge_checks=[]
    for mode in ('empty_reference','full_reference','triangle_free_support'):
        types=np.array([0,0,1,1,2]);schema=list(itertools.combinations_with_replacement(range(3),2));q=np.array(list(itertools.combinations(range(5),2)) if mode!='triangle_free_support' else [(0,1),(1,2),(2,3),(3,4)]).T
        fam=np.array([schema.index(tuple(sorted(types[x]))) for x in q.T]);ref=np.full(len(fam),int(mode=='full_reference'),dtype=np.uint8);core=HardScore(types,q,fam,ref,len(schema));calc=ExactHardMoments(core)
        p=np.linspace(.15,.85,len(fam));bits=np.array(list(itertools.product([0,1],repeat=len(fam))),dtype=np.uint8);mass=np.prod(np.where(bits,p,1-p),axis=1);cost=core.batch(bits);V,G,_=calc.value_gradient(p)
        np.testing.assert_allclose(V,mass@cost[:,[0,2,3]],atol=1e-11,rtol=1e-10);np.testing.assert_allclose(G,(cost[:,[0,2,3]]*mass[:,None]).T@(bits-p),atol=1e-11,rtol=1e-10)
        np.testing.assert_allclose(remaining_costs(core,bits),cost[:,[1,4]],atol=1e-14,rtol=1e-14);edge_checks.append(dict(mode=mode,states=len(bits),passed=True))
    assets=json.loads((CPU/'assets.json').read_text());oldbindings=json.loads((CPU/'bindings.json').read_text());cache={};timings=[];maxerr=0;maxgrad=0
    frozen=json.loads((OUT/'bindings.json').read_text())
    for p,h in frozen.items():assert sha(p)==h,'frozen asset or source changed '+p
    for a in assets:
        folder=Path(a['prediction']).parent
        for path in (a['data'],str(folder/'evaluation_inputs.pt'),a['prediction']):assert sha(path)==oldbindings[path],'historical binding changed '+path
        ds,seed,t=a['dataset'],a['seed'],a['t'];inp=next(x for x in torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs'] if x['t']==t);order=np.argsort(inp['ids'])
        if ds not in cache:
            d=np.load(a['data']);cache[ds]=HardScore(d['types'],inp['query'][:,order],inp['f'][order],inp['labels'][order],len(d['schema']))
        core=cache[ds];z=np.load(CPU/f'{ds}_{seed}_t{t}.npz');exact=np.load(OUT/f'{ds}_{seed}_t{t}.npz')
        for rep in range(4):
            ts=time.perf_counter();cost=remaining_costs(core,z['bits'][rep]);remaining_s=time.perf_counter()-ts
            np.testing.assert_array_equal(cost,z['costs'][rep][:,[1,4]]);maxerr=max(maxerr,float(abs(cost-z['costs'][rep][:,[1,4]]).max()))
            ts=time.perf_counter();grad=loo(z['probability'],z['bits'][rep],cost);loop_s=time.perf_counter()-ts
            np.testing.assert_allclose(grad,z['logit_gradients'][rep][[1,4]],atol=1e-14,rtol=1e-12);maxgrad=max(maxgrad,float(abs(grad-z['logit_gradients'][rep][[1,4]]).max()))
            np.testing.assert_array_equal(exact['hybrid_logit_gradients'][rep][[1,4]],z['logit_gradients'][rep][[1,4]])
            timings.append(dict(dataset=ds,seed=seed,t=t,repeat=rep,remaining_score_seconds=remaining_s,remaining_LOO_seconds=loop_s))
    write('remaining_cost_timings.csv',timings)
    cost=records('costs.csv');pairs=records('repeat_consistency.csv');comparison=records('estimator_comparison.csv');oldtim=list(csv.DictReader((CPU/'batches.csv').open()));summary=[];bycase=[]
    for ds in ('IMDB','PubMed'):
        cs=[x for x in cost if x['dataset']==ds];ps=[x for x in pairs if x['dataset']==ds];tt=[x for x in timings if x['dataset']==ds];ot=[x for x in oldtim if x['dataset']==ds]
        exact_s=float(np.mean([float(x['exact_seconds']) for x in cs]));remain=float(np.mean([x['remaining_score_seconds']+x['remaining_LOO_seconds'] for x in tt]));draw=float(np.mean([float(x['sampling_seconds']) for x in ot]));previous=float(np.mean([float(x['score_and_feedback_seconds']) for x in ot]));assembled=exact_s+remain+draw
        summary.append(dict(dataset=ds,conditions=len(cs),exact_three_seconds=exact_s,sampled_two_seconds=remain,reused_historical_draw_seconds=draw,assembled_five_seconds=assembled,previous_full_sampled_seconds=previous,estimated_scoring_speed_ratio=previous/assembled,old_repeat_cosine=float(np.mean([float(x['old_mean_cosine']) for x in ps])),hybrid_repeat_cosine=float(np.mean([float(x['hybrid_mean_cosine']) for x in ps])),min_hybrid_repeat_cosine=min(float(x['hybrid_mean_cosine']) for x in ps)))
    for a in assets:
        ps=[x for x in pairs if x['dataset']==a['dataset'] and int(x['seed'])==a['seed'] and int(x['t'])==a['t']]
        bycase.append(dict(dataset=a['dataset'],seed=a['seed'],t=a['t'],old_cosine=float(np.mean([float(x['old_mean_cosine']) for x in ps])),hybrid_cosine=float(np.mean([float(x['hybrid_mean_cosine']) for x in ps])),min_hybrid_cosine=min(float(x['hybrid_mean_cosine']) for x in ps)))
    write('summary.csv',summary);write('case_summary.csv',bycase)
    audit=dict(passed=True,edge_cases=edge_checks,remaining_real_scores_checked=6144,max_cost_difference=maxerr,max_remaining_gradient_difference=maxgrad,historical_bindings_unchanged=True,seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
    dump('audit.json',audit);dump('summary.json',dict(summary=summary,cases=bycase,audit=audit,timing_note='assembled exact3 + remaining2 scoring/LOO + historical draw; not full network training benchmark; setup and audit excluded'))
    print(json.dumps(dict(summary=summary,audit=audit),indent=2))
if __name__=='__main__':main()
