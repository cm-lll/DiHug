"""Descriptive summaries of the frozen CPU metric comparison, no selection."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/heat_increment_cpu_20261113'
MODES=('original','increment','amplitude_control')

def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    rows=list(csv.DictReader((OUT/'states.csv').open()))
    ev=[r for r in rows if r['panel']=='evaluation'];models=[];pairs=[];groups=[];paths=[]
    for ds,seed,arm in sorted({(r['dataset'],r['train_seed'],r['route']) for r in ev if not r['route'].endswith('_transport')}):
        before={r['sample_seed']:r for r in ev if (r['dataset'],r['train_seed'],r['route'])==(ds,seed,arm)}
        after={r['sample_seed']:r for r in ev if (r['dataset'],r['train_seed'],r['route'])==(ds,seed,arm+'_transport')}
        assert set(before)==set(after) and len(before)==4
        m=dict(dataset=ds,train_seed=seed,arm=arm,samples=4)
        keys=[mode+'_'+k for mode in MODES for k in ('score','degree','heat')]+['triangle_error','degree_W1','joint_TV','MP3_error','isolates','edges','LCC']
        for k in keys:
            bv=[float(r[k]) for r in before.values()];av=[float(r[k]) for r in after.values()]
            m[k+'_before']=mean(bv);m[k+'_after']=mean(av)
            m[k+'_before_sd']=stdev(bv);m[k+'_after_sd']=stdev(av)
            m[k+'_change_pct']=(mean(av)/mean(bv)-1)*100 if mean(bv)>1e-12 else None
        models.append(m)
        for ss in before:
            b,a=before[ss],after[ss]
            pairs.append(dict(dataset=ds,train_seed=seed,arm=arm,sample_seed=ss,
                **{k+'_delta':float(a[k])-float(b[k]) for k in keys}))
    for ds,panel,route in sorted({(r['dataset'],r['panel'],r['route']) for r in rows if r['panel']!='evaluation'}):
        rr=[r for r in rows if (r['dataset'],r['panel'],r['route'])==(ds,panel,route)]
        g=dict(dataset=ds,panel=panel,route=route,n=len(rr))
        for mode in MODES:
            for k in ('score','degree','heat'):
                v=[float(r[mode+'_'+k]) for r in rr]
                g[mode+'_'+k+'_mean']=mean(v)
            g[mode+'_score_min']=min(float(r[mode+'_score']) for r in rr)
        groups.append(g)
    for ds,seed in sorted({(r['dataset'],r['perturb_seed']) for r in rows if r['route']=='isolation_path'}):
        rr=sorted([r for r in rows if r['route']=='isolation_path' and (r['dataset'],r['perturb_seed'])==(ds,seed)],key=lambda r:int(r['level']))
        p=dict(dataset=ds,perturb_seed=seed,levels=[int(r['level']) for r in rr])
        for mode in MODES:
            v=[float(r[mode+'_score']) for r in rr];p[mode]=v;p[mode+'_monotone']=all(y>x for x,y in zip(v,v[1:]))
        paths.append(p)
    decisions=[]
    for ds in ('IMDB','PubMed'):
        pp=[r for r in pairs if r['dataset']==ds]
        dec=dict(dataset=ds,pairs=len(pp))
        for mode in MODES:
            dec[mode+'_score_better_triangle_worse']=sum(r[mode+'_score_delta']<0 and r['triangle_error_delta']>0 for r in pp)
            dec[mode+'_score_better_degreeW1_worse']=sum(r[mode+'_score_delta']<0 and r['degree_W1_delta']>0 for r in pp)
            dec[mode+'_score_better_MP3_worse']=sum(r[mode+'_score_delta']<0 and r['MP3_error_delta']>0 for r in pp)
        decisions.append(dec)
    ts=list(csv.DictReader((OUT/'time_contributions.csv').open())); attribution=[]
    for ds,seed,arm,mode in sorted({(r['dataset'],r['train_seed'],r['route'],r['mode']) for r in ts}):
        rr=[r for r in ts if (r['dataset'],r['train_seed'],r['route'],r['mode'])==(ds,seed,arm,mode)]
        total=sum(float(r['heat_contribution']) for r in rr)
        attribution.append(dict(dataset=ds,train_seed=seed,arm=arm,mode=mode,mean_heat=total/4,
            pooled_late4_heat_share=sum(float(r['heat_contribution']) for r in rr if int(r['scale_index'])>=5)/total,
            pooled_isolate_heat_share=sum(float(r['isolate_contribution']) for r in rr)/total))
    for name,rr in [('models.csv',models),('pairs.csv',pairs),('groups.csv',groups),('attribution.csv',attribution)]:table(name,rr)
    (OUT/'summary.json').write_text(json.dumps(dict(models=models,groups=groups,isolation_paths=paths,tradeoff_counts=decisions),indent=2))
    print(json.dumps(dict(models=[{k:v for k,v in m.items() if k in ('dataset','train_seed','arm','original_score_change_pct','increment_score_change_pct','amplitude_control_score_change_pct')} for m in models],paths=paths,tradeoffs=decisions),indent=2))

if __name__=='__main__':main()
