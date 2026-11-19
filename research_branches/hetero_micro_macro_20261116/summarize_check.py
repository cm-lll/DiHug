"""CPU-only summary of the bounded model check; preserves all time/type rows."""
import json,csv,statistics,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hetero_micro_macro_model_check_20261116'
def mean(x):return statistics.mean(x)
def metrics(rows,ds):
    out={k:mean(r[k] for r in rows) for k in ['CE','positive_CE','negative_CE','macro']}
    for b in ['degree','closure','propagation']:
        out[b]=mean(x['error'] for r in rows for x in r['errors'] if x['block']==b)
    for k in ['expected_edges','expected_triangles','low_fraction_error','hub_fraction_error','joint_expected_degree_proxy']:
        out[k]=mean(r['structure'][k] for r in rows)
    target=45 if ds=='IMDB' else 1937
    out['triangle_absolute_error']=mean(abs(r['structure']['expected_triangles']-target) for r in rows)
    return out
def main():
    p=json.loads((OUT/'protocol.json').read_text());summary=[];times=[];relations=[];types=[]
    for job in p['jobs']:
        path=OUT/(job+'.json')
        if not path.exists():continue
        r=json.loads(path.read_text());base=metrics(r['baseline'],r['dataset'])
        arms={'baseline':r['baseline'],**{a:v['validation'] for a,v in r['arms'].items()}}
        for arm,rows in arms.items():
            vals=metrics(rows,r['dataset'])
            summary.append(dict(job=job,dataset=r['dataset'],seed=r['seed'],arm=arm,**vals,
                relative_percent={k:100*(v-base[k])/abs(base[k]) if base[k] else None for k,v in vals.items()}))
            for x in rows:
                shared=dict(job=job,dataset=r['dataset'],seed=r['seed'],arm=arm,t=x['t'])
                times.append(dict(**shared,CE=x['CE'],positive_CE=x['positive_CE'],negative_CE=x['negative_CE'],macro=x['macro'],**{k:v for k,v in x['structure'].items() if isinstance(v,(int,float))}))
                for y in x['relations']:relations.append(dict(**shared,**y))
                for y in x['errors']:types.append(dict(**shared,**y))
    for name,rows in [('summary',summary),('per_time',times),('per_relation',relations),('per_type_block',types)]:
        (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
        if rows and name!='summary':
            with (OUT/(name+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    paired=[]
    for job in p['jobs']:
        selected={r['arm']:r for r in summary if r['job']==job}
        if not all(k in selected for k in ['baseline','CE','micro_macro']):continue
        a,b=selected['micro_macro'],selected['CE']
        paired.append(dict(job=job,seed=a['seed'],relative_candidate_vs_CE={k:100*(a[k]-b[k])/abs(b[k]) if b[k] else None for k in a if isinstance(a[k],float)}))
    (OUT/'candidate_vs_CE.json').write_text(json.dumps(paired,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
