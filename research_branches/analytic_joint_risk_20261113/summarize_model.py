"""Summarize existing model-check outputs; CPU only, no model load."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/analytic_joint_model_check_20261113'

def table(name,rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
    if rows:
        with (OUT/(name+'.csv')).open('w') as h:
            w=csv.DictWriter(h,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

details=[];relations=[];summary=[];vectors=[]
for file in sorted(OUT.glob('*_A_transport.json')):
    data=json.loads(file.read_text());ds=data['dataset'];seed=data['seed']
    base={(r['split'],r['t']):r for r in data['baseline']}
    for arm,run in data['arms'].items():
        for r in run['validation']:
            b=base[r['split'],r['t']]
            item=dict(dataset=ds,train_seed=seed,arm=arm,split=r['split'],t=r['t'])
            for key in ('risk','bias','variance'):
                x=b['loss'][key];y=r['loss'][key]
                item[key+'_before']=x;item[key+'_after']=y;item[key+'_relative_change']=y/x-1 if x else None
            for key in ('CE','positive_CE','negative_CE'):
                item[key+'_before']=b[key];item[key+'_after']=r[key]
            for key in ('joint_expected_degree_proxy','low_fraction_error','hub_fraction_error','expected_edges','expected_triangles'):
                item[key+'_before']=b['structure'][key];item[key+'_after']=r['structure'][key]
            target=45 if ds=='IMDB' else 1937
            for phase,row in [('before',b),('after',r)]:
                item['expected_triangle_abs_error_'+phase]=abs(row['structure']['expected_triangles']-target)
            details.append(item)
            for br,rr in zip(b['relations'],r['relations']):
                assert br['relation']==rr['relation']
                rel={k:item[k] for k in ('dataset','train_seed','arm','split','t')}
                rel.update(relation=rr['relation'],positive_n=rr['positive_n'],negative_n=rr['negative_n'])
                for metric in ('positive_CE','negative_CE','missed_mass','false_mass'):
                    rel[metric+'_before']=br[metric];rel[metric+'_after']=rr[metric]
                relations.append(rel)
        for split in ('update','independent'):
            selected=[r for r in details if r['dataset']==ds and r['train_seed']==seed and r['arm']==arm and r['split']==split]
            item=dict(dataset=ds,train_seed=seed,arm=arm,split=split,times=len(selected),gradient_norm=run['gradient_norm'],clip_scale=run['clip_scale'])
            for metric in ('risk','joint_expected_degree_proxy','low_fraction_error','hub_fraction_error','expected_triangle_abs_error','positive_CE','negative_CE'):
                before=sum(r[metric+'_before'] for r in selected)/len(selected);after=sum(r[metric+'_after'] for r in selected)/len(selected)
                item[metric+'_before_mean']=before;item[metric+'_after_mean']=after
                item[metric+'_change_percent_of_mean']=100*(after/before-1) if before else None
                item[metric+'_improved_times']=sum(r[metric+'_after']<r[metric+'_before'] for r in selected)
            summary.append(item)
    vectors.extend([dict(dataset=ds,train_seed=seed,**r) for r in data['comparisons']])
table('readouts',details);table('relations',relations);table('summary',summary);table('vectors',vectors)
print(json.dumps([r for r in summary if r['split']=='independent'],indent=2))
