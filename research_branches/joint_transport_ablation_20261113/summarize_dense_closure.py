"""CPU summaries of frozen old graph audit; sample SD within each model."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/dense_closure_audit_20261113'
def rows(path):return list(csv.DictReader(Path(path).open()))
def table(name,rr):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def avg(rr,k):
    x=[float(r[k]) for r in rr if r.get(k) not in ('',None)]
    return (float(np.mean(x)),float(np.std(x,ddof=1)) if len(x)>1 else None,len(x)) if x else (None,None,0)
def main():
    gg=rows(OUT/'graphs.csv');pp=rows(OUT/'paired_changes.csv');nn=rows(OUT/'nodes.csv');cc=rows(OUT/'triangle_composition.csv');pc=rows(OUT/'paired_triangle_composition.csv')
    score=rows(ROOT/'results/robust_heat_transport_20261113/old_graphs_rescored.csv');lookup={(r['dataset'],r['train_seed'],r['arm'],r['sample_seed']):r for r in score}
    for r in gg:
        if r['panel']=='evaluation':
            s=lookup[(r['dataset'],r['seed'],r['arm'],r['sample_seed'])]
            assert abs(float(r['original_score'])-float(s['original']))<1e-8
            r.update(robust_score=s['robust'],robust_degree=s['robust_degree'],robust_heat=s['robust_heat'])
    table('graphs_with_rescore.csv',gg)
    models=[];pairsummary=[];type_summary=[];typed_summary=[]
    keys=sorted(set((r['dataset'],r['seed'],r['arm']) for r in gg if r['panel']=='evaluation'))
    fields=['edges','triangles','triangle_error','wedges','transitivity','isolates','low_count','hub_count','max_degree','hub_triangle_participation_fraction','triangles_all_hub','degree_opportunity_component','closure_component','original_score','robust_score','robust_degree','robust_heat']
    for ds,seed,arm in keys:
        rr=[r for r in gg if r['panel']=='evaluation' and (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)];assert len(rr)==4
        row=dict(dataset=ds,seed=seed,arm=arm,samples=4)
        for k in fields:
            mu,sd,n=avg(rr,k);row[k+'_mean']=mu;row[k+'_sd']=sd;row[k+'_n']=n
        models.append(row)
        rr=[r for r in pp if (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)]
        if rr:
            row=dict(dataset=ds,seed=seed,arm=arm,samples=4)
            for k in rr[0]:
                if k not in ('dataset','seed','arm','sample_seed'):row[k+'_mean'],row[k+'_sd'],_=avg(rr,k)
            row['CN_enriched_pairs']=sum(float(r['added_edges_parent_CN_mean'])>float(r['relation_matched_nonedge_CN_mean']) for r in rr)
            row['hub_pair_enriched_pairs']=sum(float(r['added_edges_parent_hub_hub_fraction'])>float(r['relation_matched_nonedge_hub_hub_fraction']) for r in rr)
            pairsummary.append(row)
        rnodes=[r for r in nn if r['panel']=='evaluation' and (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)]
        for typ in sorted(set(r['type'] for r in rnodes)):
            for group in ['low','middle','hub']:
                group_rows=[r for r in rnodes if r['type']==typ and r['group']==group]
                type_summary.append(dict(dataset=ds,seed=seed,arm=arm,type=typ,group=group,node_count_per_graph=len(group_rows)/4,
                    triangle_participations_per_graph=sum(int(r['triangles']) for r in group_rows)/4,
                    degree_sum_per_graph=sum(int(r['degree']) for r in group_rows)/4))
        rcomp=[r for r in cc if r['panel']=='evaluation' and (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)]
        for composition in sorted(set(r['composition'] for r in rcomp)):
            rr=[r for r in rcomp if r['composition']==composition];mu,sd,n=avg(rr,'triangles');typed_summary.append(dict(dataset=ds,seed=seed,arm=arm,composition=composition,mean=mu,sd=sd,n=n))
    for name,rr in [('model_summary.csv',models),('paired_summary.csv',pairsummary),('type_group_summary.csv',type_summary),('typed_triangle_summary.csv',typed_summary)]:table(name,rr)
    # Reward associations are four-graph descriptive batches, not independent model replicates.
    batches=[]
    keys=sorted(set((r['dataset'],r['seed'],r['arm'],r['update']) for r in gg if r['panel']=='training'))
    for key in keys:
        rr=[r for r in gg if r['panel']=='training' and (r['dataset'],r['seed'],r['arm'],r['update'])==key];assert len(rr)==4
        reward=np.array([float(r['reward_advantage']) for r in rr]);row=dict(zip(('dataset','seed','arm','update'),key))
        for k in ['edges','isolates','low_count','triangles','triangle_error','wedges','transitivity','triangles_all_hub']:
            v=np.array([float(r[k]) for r in rr]);row[k+'_reward_corr']=float(np.corrcoef(v,reward)[0,1]) if np.std(v)>0 and np.std(reward)>0 else None
        batches.append(row)
    table('training_reward_associations.csv',batches)
    (OUT/'summary.json').write_text(json.dumps(dict(models=models,pairs=pairsummary),indent=2))
    for m in models:
        if m['arm'] in ['A','A_transport']:print({k:v for k,v in m.items() if k in ['dataset','seed','arm','edges_mean','edges_sd','triangles_mean','triangles_sd','transitivity_mean','transitivity_sd','isolates_mean','robust_score_mean']})
if __name__=='__main__':main()
