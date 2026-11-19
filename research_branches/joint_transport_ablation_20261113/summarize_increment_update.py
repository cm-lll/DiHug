"""CPU readout of saved raw probabilities and temporary-update records."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/increment_update_check_20261113'
ARMS=('original','increment','amplitude_control')

def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def rawmetrics(v):
    m={k:float(v[k]) for k in ('CE','positive_CE','negative_CE','node_total_degree_MAE','expected_edge_error','expected_edges')}
    for k in ('total_degree_W1','joint_relation_degree_L1','marginal_relation_degree_W1','top10_degree_mean_error'):
        m[k]=float(np.mean([x[k] for x in v['anonymous']]))
    m['reference_hub_degree_MAE']=float(np.mean([x['hub_MAE'] for x in v['type_groups'] if x['hub_MAE'] is not None]))
    m['reference_low_degree_MAE']=float(np.mean([x['low_MAE'] for x in v['type_groups'] if x['low_MAE'] is not None]))
    return m

def product_proxies(z,t):
    p=z['p_'+str(t)];q=z['query_'+str(t)];lab=z['label_'+str(t)];n=len(z['types'])
    assert np.all((p>=0)&(p<=1)) and len(set(map(tuple,q.T)))==len(p)
    a=np.zeros((n,n));ref=np.zeros_like(a);a[q[0],q[1]]=p;a[q[1],q[0]]=p
    ref[q[0],q[1]]=lab;ref[q[1],q[0]]=lab
    iso=np.prod(1-a,axis=1);refiso=int((ref.sum(1)==0).sum())
    tri=float(np.sum(a*(a@a))/6);reftri=float(np.sum(ref*(ref@ref))/6)
    return dict(product_expected_isolates=float(iso.sum()),product_expected_isolate_count_error=abs(float(iso.sum())-refiso),
                product_expected_triangles=tri,reference_triangles=reftri,
                product_expected_triangle_count_error=abs(tri-reftri))

def main():
    start=time.time();done=json.loads((OUT/'completion.json').read_text());assert done['status']=='complete'
    rows=[];rels=[];groups=[];update=[];vectors=[];paired=[];models=[]
    for job in done['conditions']:
        x=json.loads((OUT/(job+'.json')).read_text());base={v['t']:v for v in x['baseline']}
        values={'baseline':x['baseline'],**{a:x['arms'][a]['validation'] for a in ARMS}}
        for arm,vv in values.items():
            z=np.load(OUT/f'{job}_{arm}_probabilities.npz')
            for v in vv:
                t=v['t'];assert v['input_hash']==base[t]['input_hash'];ident=dict(dataset=x['dataset'],seed=x['seed'],arm=arm,t=t,input_hash=v['input_hash'])
                rows.append({**ident,**rawmetrics(v),**product_proxies(z,t)})
                rels.extend({**ident,**r} for r in v['relations'])
                groups.extend({**ident,**r} for r in v['anonymous'])
        for arm,r in x['arms'].items():
            update.append(dict(dataset=x['dataset'],seed=x['seed'],arm=arm,gradient_norm=r['gradient_norm_before_clip'],
                clip_scale=r['clip_scale'],seconds=r['update_seconds'],EMA_updates=r['EMA_updates'],RNG_preserved=r['RNG_preserved']))
        vectors.extend(dict(dataset=x['dataset'],seed=x['seed'],**r) for r in x['comparisons'])
    metrics=[k for k in rows[0] if k not in ('dataset','seed','arm','t','input_hash','reference_triangles')]
    for r in rows:
        if r['arm']=='baseline':continue
        b=next(x for x in rows if x['arm']=='baseline' and all(x[k]==r[k] for k in ('dataset','seed','t')))
        o=next(x for x in rows if x['arm']=='original' and all(x[k]==r[k] for k in ('dataset','seed','t')))
        paired.append({k:r[k] for k in ('dataset','seed','arm','t')}|{k+'_delta_baseline':r[k]-b[k] for k in metrics}|{k+'_delta_original':r[k]-o[k] for k in metrics})
    for ds,seed in sorted({(r['dataset'],r['seed']) for r in rows}):
        for arm in ('baseline',)+ARMS:
            rr=[r for r in rows if (r['dataset'],r['seed'],r['arm'])==(ds,seed,arm)];assert len(rr)==6
            m=dict(dataset=ds,seed=seed,arm=arm,inputs=6)
            for k in metrics:m[k]=float(np.mean([r[k] for r in rr]))
            for k in ('CE','positive_CE','negative_CE','joint_relation_degree_L1','top10_degree_mean_error'):
                for tag,ts in [('low',(1,75)),('middle',(250,550)),('high',(800,1000))]:
                    m[k+'_'+tag]=float(np.mean([r[k] for r in rr if r['t'] in ts]))
            models.append(m)
    for name,rr in [('readouts.csv',rows),('relations.csv',rels),('anonymous_types.csv',groups),('updates.csv',update),('vectors.csv',vectors),('paired_readouts.csv',paired),('model_summary.csv',models)]:table(name,rr)
    (OUT/'evaluation_scope.json').write_text(json.dumps(dict(readout='raw only',input_panel='previously viewed, same inputs across arms',
      main_metrics='mean over six fixed times, equal node-type mean for anonymous degree metrics; keep individual times',
      product_proxies='Independent Bernoulli edges conditional on each clean-head output; sum product(1-p) gives expected isolates, tr(P^3)/6 expected triangles. Not reverse-chain graph metrics. Absolute error of expected count, NOT expected absolute error.',
      reports='Reference-ID low/hub MAE separate from anonymous metrics; no thresholding soft probabilities into degrees or graphs',
      seconds=time.time()-start,new_graphs=0),indent=2))
    print(json.dumps([ {k:r[k] for k in ('dataset','seed','arm','CE','joint_relation_degree_L1','top10_degree_mean_error','product_expected_isolates','product_expected_triangles')} for r in models],indent=2))

if __name__=='__main__':main()
