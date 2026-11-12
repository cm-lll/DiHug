"""One frozen CPU-only joint relation-degree + typed heat kernel audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import csv,json,time,hashlib,resource,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.stats import wasserstein_distance,spearmanr

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/typed_heat_audit_20261111'
OUT=ROOT/'results/typed_heat_degree_audit_20261111'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def table(name,rows):
    save(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def degrees(bits,q,f,n,r):
    d=np.zeros((n,r),dtype=np.int64);which=bits.astype(bool)
    np.add.at(d,(q[0,which],f[which]),1);np.add.at(d,(q[1,which],f[which]),1)
    assert d.sum()==2*int(bits.sum())
    return d
def median_bandwidth(x):
    v=pdist(x,'sqeuclidean');v=v[v>1e-24]
    return float(np.median(v)) if len(v) else 1.0
def kernel(x,y,s):return np.exp(-cdist(x,y,'sqeuclidean')/(2*s))
def mmd(a,b,c):return float((a.mean()+b.mean()-2*c.mean())/2)

def checks():
    h=np.array([[0.],[1.]]);d=h.copy();swapped=d[::-1].copy()
    k=kernel(h,h,1);kd=kernel(swapped,swapped,1);cross=kernel(swapped,d,1)
    separate_heat=mmd(k,k,k);separate_degree=mmd(kd,k,cross)
    joint=mmd(k*kd,k*k,k*cross)
    assert abs(separate_heat)<1e-12 and abs(separate_degree)<1e-12 and joint>0
    x=np.column_stack([h,swapped]);y=np.column_stack([h,d])
    direct=mmd(kernel(x,x,1),kernel(y,y,1),kernel(x,y,1))
    assert abs(joint-direct)<1e-12
    return dict(joint_coupling_fixture=joint,separate_heat=separate_heat,separate_degree=separate_degree,
                concatenated_kernel_difference=abs(joint-direct),meaning='joint-versus-marginal algebra check, not a new scientific graph')

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=False)
    oldrows=json.loads((SRC/'per_graph.json').read_text());assets=json.loads((SRC/'assets.json').read_text())
    oldprotocol=json.loads((SRC/'protocol.json').read_text())
    for p,h in oldprotocol['bindings'].items():assert sha(p)==h
    bind={str(SRC/k):sha(SRC/k) for k in ['per_graph.json','assets.json','protocol.json','signatures.npz']}
    bind[str(Path(__file__).resolve())]=sha(__file__)
    save('protocol.json',dict(frozen_before_new_scores=True,source=str(SRC),graphs=70,new_graphs=0,
        candidate='Per-type MMD2/2 with product k_heat*k_degree, then equal mean of types; one candidate only',
        heat='Cached unchanged 45-d features, scales, reference heat bandwidths, and graphs',
        degree='Per-node 10-relation degree; log1p(degree)/log1p(max(1,reference per-type relation mean)), divided by sqrt(10)',
        degree_bandwidth='Reference within-type median positive squared distance (>1e-24); fallback 1; reference only',
        coupling='Product kernel adds dimensionless squared distances with coefficient 1 each; preserves original heat kernel exactly; not sum of two marginal losses',
        diagnostics='Degree-only MMD attribution (not competing candidate); type-specific tail statistics and independent joint coupling fixture',
        tails='Low degree <=1; hubs >=max(2,ceil(reference type total-degree q90)); top ceil(10%*type_size) degree mass; no forced equal group sizes',
        rules='No tuning based on scores; no claims that every CE graph must outrank every perturbation; no training readiness inferred from ranking',
        compute='One CPU thread, 600-second limit, no network model/GPU/updates/new samples',bindings=bind))
    tests=checks();cache=np.load(SRC/'signatures.npz');meta={};stored={};group_rows=[];rows=[]
    tasks=json.loads((ROOT/'results/real192_update_audit_20261111/protocol.json').read_text())['tasks']
    for asset in assets:
        dataset=asset['dataset'];task=next(t for t in tasks if t['parent']['dataset']==dataset)
        data=np.load(task['parent']['data']);types=data['types'];r=len(data['schema'])
        z=np.load(ROOT/'results/real192_update_audit_20261111'/task['id']/'validation_u2000_raw_t1.npz')
        refrow=next(v for v in oldrows if v['dataset']==dataset and v['arm']=='reference');key=refrow['signature_key']
        dr=degrees(cache[key+'_bits'],z['query'],z['families'],len(types),r)
        hr=cache[key+'_features'];scales={};bands={};thresholds={};ref_features={}
        for t in np.unique(types):
            select=types==t;scales[int(t)]=np.log1p(np.maximum(1,dr[select].mean(0)))
            v=np.log1p(dr[select])/scales[int(t)]/np.sqrt(r);ref_features[int(t)]=v
            bands[int(t)]=median_bandwidth(v)
            thresholds[int(t)]=max(2,int(np.ceil(np.quantile(dr[select].sum(1),.9))))
        meta[dataset]=dict(types=types,query=z['query'],families=z['families'],r=r,ref_degree=dr,
            ref_heat=hr,scales=scales,bands=bands,ref_features=ref_features,thresholds=thresholds,
            heat_bands={int(k):v for k,v in asset['bandwidth_squared'].items()})
    save('reference_scaling.json',{k:dict(log_degree_scales={str(t):a.tolist() for t,a in v['scales'].items()},
        degree_bandwidth_squared=v['bands'],heat_bandwidth_squared=v['heat_bands'],hub_thresholds=v['thresholds']) for k,v in meta.items()})
    max_old_error=0;permutation_error=0
    for index,row in enumerate(oldrows):
        if time.time()-started>600:raise TimeoutError('CPU internal deadline')
        tick=time.perf_counter();dataset=row['dataset'];m=meta[dataset];ty=m['types'];key=row['signature_key']
        d=degrees(cache[key+'_bits'],m['query'],m['families'],len(ty),m['r']);h=cache[key+'_features']
        base=[];only_degree=[];joint=[];feature=np.zeros_like(d,dtype=float)
        for t in np.unique(ty):
            t=int(t);select=ty==t;x=h[select];y=m['ref_heat'][select]
            dx=np.log1p(d[select])/m['scales'][t]/np.sqrt(m['r']);dy=m['ref_features'][t];feature[select]=dx
            a=kernel(x,x,m['heat_bands'][t]);b=kernel(y,y,m['heat_bands'][t]);c=kernel(x,y,m['heat_bands'][t])
            ad=kernel(dx,dx,m['bands'][t]);bd=kernel(dy,dy,m['bands'][t]);cd=kernel(dx,dy,m['bands'][t])
            base.append(mmd(a,b,c));only_degree.append(mmd(ad,bd,cd));joint.append(mmd(a*ad,b*bd,c*cd))
            # Independent concatenated feature calculation; paired feature rows preserved.
            xx=np.column_stack([x/np.sqrt(m['heat_bands'][t]),dx/np.sqrt(m['bands'][t])])
            yy=np.column_stack([y/np.sqrt(m['heat_bands'][t]),dy/np.sqrt(m['bands'][t])])
            check=mmd(kernel(xx,xx,1),kernel(yy,yy,1),kernel(xx,yy,1))
            assert abs(check-joint[-1])<1e-11
            reverse=xx[::-1];checkperm=mmd(kernel(reverse,reverse,1),kernel(yy,yy,1),kernel(reverse,yy,1))
            permutation_error=max(permutation_error,abs(checkperm-check));assert abs(checkperm-check)<1e-11
            total=d[select].sum(1);reference=m['ref_degree'][select].sum(1);threshold=m['thresholds'][t];top=max(1,int(np.ceil(.1*len(total))))
            fraction=lambda v:float(np.sort(v)[-top:].sum()/v.sum()) if v.sum()>0 else None
            refshare=fraction(reference);share=fraction(total)
            group_rows.append(dict(dataset=dataset,task=row['task'],arm=row['arm'],signature_key=key,type=t,nodes=len(total),
                heat_cost=base[-1],degree_only_cost=only_degree[-1],joint_cost=joint[-1],
                total_degree_W1=float(wasserstein_distance(total,reference)),
                low_count=int((total<=1).sum()),reference_low_count=int((reference<=1).sum()),
                low_fraction_error=abs(float((total<=1).mean()-(reference<=1).mean())),
                hub_threshold=threshold,hub_count=int((total>=threshold).sum()),reference_hub_count=int((reference>=threshold).sum()),
                hub_fraction_error=abs(float((total>=threshold).mean()-(reference>=threshold).mean())),
                max_degree=int(total.max()),reference_max_degree=int(reference.max()),max_degree_abs_error=int(abs(total.max()-reference.max())),
                top10_degree_mass=share,reference_top10_degree_mass=refshare,
                top10_mass_abs_error=abs(share-refshare) if share is not None and refshare is not None else None))
        error=abs(float(np.mean(base))-row['typed_heat_cost']);max_old_error=max(max_old_error,error);assert error<1e-11
        assert min(joint)>-1e-10
        if row['arm']=='reference':assert abs(np.mean(joint))<1e-12
        rec=dict(row);rec.update(joint_heat_degree_cost=float(np.mean(joint)),degree_only_cost=float(np.mean(only_degree)),
            new_group_costs=json.dumps(joint),new_scoring_seconds=time.perf_counter()-tick)
        rows.append(rec);stored[key+'_degree_features']=feature
    table('per_graph',rows);table('per_type',group_rows);np.savez_compressed(OUT/'degree_features.npz',**stored)
    summary=[]
    for dataset in meta:
        for task,arm in sorted(set((r['task'],r['arm']) for r in rows if r['dataset']==dataset)):
            rr=[r for r in rows if r['dataset']==dataset and r['task']==task and r['arm']==arm]
            rec=dict(dataset=dataset,task=task,arm=arm,n=len(rr))
            for metric in ['typed_heat_cost','joint_heat_degree_cost','degree_only_cost']:
                rec[metric+'_mean']=float(np.mean([r[metric] for r in rr]));rec[metric+'_sd']=float(np.std([r[metric] for r in rr],ddof=1)) if len(rr)>1 else None
            summary.append(rec)
    table('summary',summary)
    correlations=[]
    for dataset in meta:
        for panel in ['all_old_graphs','ordinary_CE_only']:
            rr=[r for r in rows if r['dataset']==dataset and r['path'] and (panel=='all_old_graphs' or r['arm']=='ordinary_CE')]
            for metric in ['degree_W1','joint_degree_cost','present_triangle_cost','absent_triangle_cost','components_cost','relation_count_cost']:
                for method in ['typed_heat_cost','joint_heat_degree_cost']:
                    x=[r[method] for r in rr];y=[r[metric] for r in rr]
                    rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
                    correlations.append(dict(dataset=dataset,panel=panel,n=len(rr),method=method,metric=metric,spearman=rho))
    table('correlations',correlations)
    tests.update(old_score_max_difference=max_old_error,within_type_permutation_max_error=permutation_error,
                 independent_kernel_checks=len(group_rows),source_graphs_unchanged=True)
    save('checks.json',tests)
    for p,h in bind.items():assert sha(p)==h
    save('receipt.json',dict(status='complete',seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        threads=1,graphs_reused=len(rows),per_type_rows=len(group_rows),new_graphs=0,heat_recomputations=0,model_calls=0,GPU_hours=0,updates=0,
        median_scoring_seconds_including_checks=float(np.median([r['new_scoring_seconds'] for r in rows]))))
    print(json.dumps({'summary':summary,'receipt':json.loads((OUT/'receipt.json').read_text())},indent=2))

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure.json',{'traceback':traceback.format_exc()})
        raise
