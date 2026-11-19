"""Summaries of all old batches; do not treat rounds/chains as training repeats."""
import csv
import json
from pathlib import Path
from statistics import mean,median
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/increment_reward_audit_20261113_v2'
MODES=('original','increment','amplitude_control')

def read(name):return list(csv.DictReader((OUT/name).open()))
def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    b=read('batches.csv');a=read('metric_feedback.csv');g=read('graphs.csv');pref=read('preferred_descriptive.csv')
    permodel=[];permetric=[];allmetrics=[];prechanges=[]
    for ds,seed,arm in sorted({(r['dataset'],r['train_seed'],r['arm']) for r in b}):
        rr=[r for r in b if (r['dataset'],r['train_seed'],r['arm'])==(ds,seed,arm)];assert len(rr)==4
        for mode in MODES:
            permodel.append(dict(dataset=ds,train_seed=seed,arm=arm,mode=mode,batches=4,
                heat_degree_adv_ratio_median=median(float(r[mode+'_heat_degree_adv_ratio']) for r in rr if r[mode+'_heat_degree_adv_ratio']),
                reward_cosine_original_mean=mean(float(r[mode+'_reward_cosine_original']) for r in rr if r[mode+'_reward_cosine_original']),
                preferred_changed=sum(r[mode+'_preferred_changed']=='True' for r in rr),
                reward_sign_flips=sum(int(r[mode+'_sign_flips']) for r in rr),
                label_vote_positive_batches=sum(float(r[mode+'_terminal_label_vote'])>0 for r in rr),
                reward_norm_ratio_median=median(float(r[mode+'_reward_norm'])/float(r['original_reward_norm']) for r in rr)))
            for metric in sorted({r['metric'] for r in a}):
                aa=[r for r in a if (r['dataset'],r['train_seed'],r['arm'],r['mode'],r['metric'])==(ds,seed,arm,mode,metric)]
                vv=[float(r['reward_metric_alignment']) for r in aa if r['reward_metric_alignment']]
                permetric.append(dict(dataset=ds,train_seed=seed,arm=arm,mode=mode,metric=metric,batches=4,valid_batches=len(vv),
                    aligned_to_larger_error=sum(v>0 for v in vv),aligned_to_smaller_error=sum(v<0 for v in vv),mean_alignment=mean(vv) if vv else None))
    for ds in ('IMDB','PubMed'):
        for mode in MODES:
            for metric in sorted({r['metric'] for r in a}):
                rr=[r for r in a if (r['dataset'],r['mode'],r['metric'])==(ds,mode,metric)]
                v=[float(r['reward_metric_alignment']) for r in rr if r['reward_metric_alignment']]
                allmetrics.append(dict(dataset=ds,mode=mode,metric=metric,batches=len(rr),valid_batches=len(v),
                    aligned_to_larger_error=sum(x>0 for x in v),mean_alignment=mean(v) if v else None))
    for r in pref:
        if r['mode']=='original':continue
        old=next(x for x in pref if x['mode']=='original' and all(x[k]==r[k] for k in ('dataset','train_seed','arm','update')))
        if r['preferred_chain']==old['preferred_chain']:continue
        out={k:r[k] for k in ('dataset','train_seed','arm','update','mode')}
        out.update(old_chain=old['preferred_chain'],new_chain=r['preferred_chain'])
        for k in ('edge_error','degree_W1','joint_relation_degree_L1','type_top10_degree_mean_error','type_low_fraction_error','isolates','LCC_error','triangle_error'):
            out[k+'_old']=float(old[k]);out[k+'_new']=float(r[k]);out[k+'_delta']=float(r[k])-float(old[k])
        prechanges.append(out)
    table('model_summary.csv',permodel);table('model_metric_summary.csv',permetric);table('dataset_metric_summary.csv',allmetrics);table('preferred_changes.csv',prechanges)
    # Keep all four candidate graphs and feedback values in the focal old batch.
    focus=[r for r in g if r['dataset']=='PubMed' and r['train_seed']=='2026110422' and r['arm']=='A' and r['update']=='1']
    table('pubmed_A0422_round1.csv',focus)
    summary=dict(batches=32,graphs=128,models=8,permodel=permodel,permetric=allmetrics,
        limits=['All historical policy trajectories; no fresh candidate trajectory',
                'Four rounds within one model are dependent', 'Correlations on four graphs descriptive only',
                'Terminal-label vote is incomplete logit feedback, not AdamW update'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps([r for r in permetric if r['arm']=='A' and r['mode'] in ('original','increment') and r['metric'] in ('joint_relation_degree_L1','type_top10_degree_mean_error','isolates','triangle_error')],indent=2))

if __name__=='__main__':main()
