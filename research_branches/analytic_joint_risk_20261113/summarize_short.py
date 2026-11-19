"""CPU-only complete and partial short-run summaries; no model execution."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/analytic_joint_short_20261113'
def table(name,rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
    if rows:
        with (OUT/(name+'.csv')).open('w') as h:
            w=csv.DictWriter(h,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
details=[];relations=[];type_rows=[];summary=[];updates=[];status=[]
for folder in sorted(OUT.glob('*_A_transport')):
    if not (folder/'result.json').exists():continue
    d=json.loads((folder/'result.json').read_text());ds=d['dataset'];seed=d['train_seed']
    base={(r['t'],r['rep']):r for r in d['baseline']};target=45 if ds=='IMDB' else 1937
    status.append(dict(job=folder.name,completed=(folder/'completion.json').exists(),failed=(folder/'failure.json').exists()))
    for arm,run in d['arms'].items():
        for u in run['updates']:
            updates.append(dict(dataset=ds,train_seed=seed,arm=arm,step=u['step'],gradient_norm=u['gradient_norm'],
                clip_scale=u['clip_scale'],parameter_delta_L2=u['parameter_delta_L2'],seconds=u['seconds'],
                risk_mean=sum(r['risk'] for r in u['loss_inputs'])/6,bias_mean=sum(r['bias'] for r in u['loss_inputs'])/6))
        for step,evals in run['evaluations'].items():
            for readout in ('raw','ema'):
                if readout not in evals:continue
                selected=[]
                for r in evals[readout]:
                    b=base[r['t'],r['rep']];assert b['input_hash']==r['input_hash']
                    x=dict(dataset=ds,train_seed=seed,arm=arm,step=int(step),readout=readout,t=r['t'],rep=r['rep'])
                    for key in ('risk','bias','variance'):
                        x[key+'_before']=b['loss'][key];x[key+'_after']=r['loss'][key]
                    for key in ('CE','positive_CE','negative_CE'):
                        x[key+'_before']=b[key];x[key+'_after']=r[key]
                    for key in ('joint_expected_degree_proxy','low_fraction_error','hub_fraction_error','expected_edges','expected_triangles'):
                        x[key+'_before']=b['structure'][key];x[key+'_after']=r['structure'][key]
                    for phase,rr in [('before',b),('after',r)]:
                        x['expected_triangle_abs_error_'+phase]=abs(rr['structure']['expected_triangles']-target)
                    details.append(x);selected.append(x)
                    for bg,ag in zip(b['structure']['types'],r['structure']['types']):
                        assert bg['type']==ag['type']
                        y={k:x[k] for k in ('dataset','train_seed','arm','step','readout','t','rep')}
                        y.update(type=ag['type'],nodes=ag['nodes'],hub_threshold=ag['hub_threshold'],
                            reference_low_fraction=ag['reference_low_fraction'],reference_hub_fraction=ag['reference_hub_fraction'])
                        for key in ('expected_low_fraction','expected_hub_fraction','expected_degree_joint_L1_proxy'):
                            y[key+'_before']=bg[key];y[key+'_after']=ag[key]
                        type_rows.append(y)
                    for br,rr in zip(b['relations'],r['relations']):
                        y={k:x[k] for k in ('dataset','train_seed','arm','step','readout','t','rep')}
                        y.update(relation=rr['relation'],positive_n=rr['positive_n'],negative_n=rr['negative_n'])
                        for key in ('positive_CE','negative_CE','missed_mass','false_mass'):
                            y[key+'_before']=br[key];y[key+'_after']=rr[key]
                        relations.append(y)
                x=dict(dataset=ds,train_seed=seed,arm=arm,step=int(step),readout=readout,inputs=len(selected))
                for key in ('risk','bias','variance','joint_expected_degree_proxy','low_fraction_error','hub_fraction_error','expected_triangle_abs_error','positive_CE','negative_CE'):
                    before=sum(r[key+'_before'] for r in selected)/len(selected);after=sum(r[key+'_after'] for r in selected)/len(selected)
                    x[key+'_before_mean']=before;x[key+'_after_mean']=after
                    x[key+'_change_percent']=100*(after/before-1) if before else None
                    x[key+'_improved_inputs']=sum(r[key+'_after']<r[key+'_before'] for r in selected)
                summary.append(x)
table('summary',summary);table('readouts',details);table('relations',relations);table('types',type_rows);table('updates',updates);table('status',status)
print(json.dumps([r for r in summary if r['step']==16 and r['readout']=='raw'],indent=2))
