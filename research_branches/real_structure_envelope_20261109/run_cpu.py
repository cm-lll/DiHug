"""Read-only archived full-graph quality/diversity audit; no acceptable-set fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,itertools,json,resource,signal,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/cdf_risk_audit_20261109/attempt2'
OUT=ROOT/'results/real_structure_envelope_20261109';START=time.monotonic();HASHES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None:assert h==expected,str(p)
    HASHES[str(p)]=h;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text())
def save(n,x):
    (OUT/(n+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(x,list) and x:
        fields=list(dict.fromkeys(k for row in x for k in row))
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(x)
def moments(values):
    v=np.array([x for x in values if x is not None],float)
    return dict(n=len(v),missing=len(values)-len(v),mean=float(v.mean()) if len(v) else None,
        SD=float(v.std(ddof=1)) if len(v)>1 else None,min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None)
def identity(j):return dict(panel=j['panel'],dataset=j['dataset'],train_seed=j['train_seed'],arm=j['arm'],sample_seed=j['sample_seed'])
def key(j):return tuple(j[k] for k in ('panel','dataset','train_seed','arm'))
def modelid(k):return '|'.join(map(str,k))
def graph_features(types,schema,edges):
    n=len(types);a=np.zeros((n,n),np.int64);a[edges[0],edges[1]]=a[edges[1],edges[0]]=1
    assert np.trace(a)==0 and a.sum()==2*edges.shape[1]
    lookup={tuple(x):r for r,x in enumerate(schema)};degrees=np.zeros((n,len(schema)),int);counts=np.zeros(len(schema),int)
    for u,v in edges.T:
        r=lookup[tuple(sorted((types[u],types[v])))];degrees[u,r]+=1;degrees[v,r]+=1;counts[r]+=1
    signature=tuple(tuple(sorted(map(tuple,degrees[types==t]))) for t in np.unique(types))
    edge_set=set((np.minimum(edges[0],edges[1])*n+np.maximum(edges[0],edges[1])).tolist())
    tri=int(np.sum((a@a)*a)//6);remaining=set(range(n));sizes=[]
    while remaining:
        stack=[remaining.pop()];size=0
        while stack:
            u=stack.pop();size+=1;new=remaining.intersection(np.flatnonzero(a[u]));remaining-=new;stack.extend(new)
        sizes.append(size)
    return dict(signature=signature,edges=edge_set,family_counts=counts,triangles=tri,lcc=max(sizes),components=len(sizes),isolates=sum(s==1 for s in sizes))
def overlap(a,b):
    c=len(a&b);union=len(a|b)
    return dict(common_edges=c,jaccard=c/union if union else None,common_over_first=c/len(a) if a else None,common_over_second=c/len(b) if b else None)
FOCUS_DISTANCE=['main_incident4_W1','all10_compat_W1','total_degree_W1','main_joint_degree_TV','triangle_composition_TV','four_cycle_composition_TV','MP1_composition_TV','MP2_composition_TV','MP3_composition_TV']
FOCUS_RAW=['edges','triangles','lcc','components','isolates','maximum_degree','four_cycles','MP1_total','MP2_total','MP3_total','average_clustering','reachable_pair_fraction']
CORE=['distance.main_incident4_W1','distance.main_joint_degree_TV','absolute_error.triangles','distance.triangle_composition_TV','distance.MP3_composition_TV','absolute_error.lcc']
def extract(m):
    ans={}
    for section in ['raw','absolute_error']:
        for k in FOCUS_RAW:ans[section+'.'+k]=m[section].get(k)
        for k,v in m[section].items():
            if k.startswith('type_') and k.endswith(('_low_le1_fraction','_zero_fraction','_maximum','_second','_p99')):ans[section+'.'+k]=v
            if k.startswith('family') and ('edges' in k):ans[section+'.'+k]=v
    for k in FOCUS_DISTANCE:ans['distance.'+k]=m['distance'].get(k)
    for k,v in m['distance'].items():
        if k.startswith(('family','type_')):ans['distance.'+k]=v
    return ans
def comparison(a,b):
    if any(a.get(k) is None or b.get(k) is None for k in CORE):return 'missing'
    diff=np.array([a[k]-b[k] for k in CORE]);tol=1e-12
    if np.all(abs(diff)<=tol):return 'equal'
    if np.all(diff<=tol):return 'dominates'
    if np.all(diff>=-tol):return 'dominated'
    return 'tradeoff'
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(300);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='All 144 previously audited full graphs / 36 model groups, 128 main and 16 separate confirmation; no candidate selection or thresholds fitted',
        reads='Graph hashes, prior metric hashes, schema/type identity, original/LCC metrics, recorded weight binding (weights not loaded or rehashed)',
        calculations='Independent hard graph counts and typed joint degree signatures; all 6 within-model pairs, all-reference overlap; signature mismatch certifies nonisomorphism; equal signatures remain unknown unless identical edges',
        quality='Per-model sample means/SD; within each panel/dataset/arm across training model means; full metric source links retained. Six-metric Pareto is descriptive, not acceptance.',
        core_metrics=CORE,tolerance='1e-12 only float comparison, not scientific acceptance tolerance',
        limits=dict(seconds=300,threads=1,memory_MiB=2048,GPU=0),
        forbidden='No threshold fitting, new score, network, training, random sample, deletion, or production modification'))
    oldhash=read(OLD/'binding.json');jobs=read(OLD/'graph_inventory.json');assert len(jobs)==144
    configs={};refs={};reference_records={};graphs={};records=[];metric_rows=[];full_index=[];inventory=[];pool=defaultdict(list);missing=[]
    data=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets'
    for ds in ['IMDB','PubMed']:
        p=data/(ds+'.npz')
        with np.load(bind(p,oldhash[str(p)])) as z:
            t=z['types'].copy();s=z['schema'].copy();e=z['edges'].copy();e=e if e.shape[0]==2 else e.T
        configs[ds]=(t,s);refs[ds]=graph_features(t,s,e)
    for i,j in enumerate(jobs):
        ident=identity(j);ds=j['dataset'];mid=modelid(key(j));uid=mid+'|'+str(j['sample_seed']);types,schema=configs[ds]
        with np.load(bind(j['file'],j['expected_sha'])) as z:
            assert np.array_equal(z['types'],types) and np.array_equal(z['schema'],schema)
            edges=z['edges'].copy();edges=edges if edges.shape[0]==2 else edges.T
        g=graph_features(types,schema,edges);graphs[uid]=g
        m=read(j['metric'],oldhash[j['metric']]);gb=m['graph_binding']
        assert m['dataset']==ds and m['train_seed']==j['train_seed'] and m['sample_seed']==j['sample_seed'] and m['updates']==3000
        assert gb['graph_sha256']==j['expected_sha'] and gb['readout']=='EMA0.99' and gb['R']==1 and gb['T']==1000 and gb['status']=='complete'
        for k in ['triangles','lcc','components','isolates']:assert g[k]==m['statistics'][k],(uid,k)
        assert len(g['edges'])==m['raw']['edges'] and np.array_equal(g['family_counts'],m['statistics']['family_edges'])
        if ds in reference_records:assert reference_records[ds]==m['reference']
        else:reference_records[ds]=m['reference']
        ref=refs[ds];ov=overlap(g['edges'],ref['edges'])
        sigsame=g['signature']==ref['signature']
        exact=g['edges']==ref['edges'];iso='identical' if exact else ('excluded_by_joint_degree' if not sigsame else 'unknown')
        vals=extract(m);pool[key(j)].append((uid,vals));rec=dict(ident,uid=uid,model=mid,**ov,reference_isomorphism=iso,**vals)
        records.append(rec)
        inventory.append(dict(j,uid=uid,model=mid,weight_sha256_recorded=gb['model_sha256'],weight_rehashed=False,
            checked_R=1,checked_T=1000,checked_updates=3000,checked_readout='EMA0.99',schema_valid=True,raw_graph_hash_verified=True))
        for version,path in [('original',Path(j['metric'])),('lcc',Path(j['metric']).with_name('lcc.json'))]:
            if not path.exists():missing.append(dict(uid=uid,asset=str(path),reason='missing'));continue
            mm=m if version=='original' else read(path,oldhash.get(str(path)))
            full_index.append(dict(ident,uid=uid,version=version,path=str(path),sha256=HASHES[str(path)],
                sections=['raw','distance','absolute_error','statistics','reference'],note='Full original fields including null retained in this unchanged file; LCC not a new sample'))
            for k,v in extract(mm).items():metric_rows.append(dict(ident,uid=uid,model=mid,version=version,metric=k,value=v))
    assert len(pool)==36 and all(len(v)==4 for v in pool.values())
    save('asset_inventory',inventory);save('full_metric_index',full_index);save('per_graph',records);save('metrics_long',metric_rows);save('references',reference_records)
    pairrows=[];pairmodels=[]
    for k,items in pool.items():
        local=[]
        for (u,_),(v,_) in itertools.combinations(items,2):
            ga,gb=graphs[u],graphs[v];same=ga['signature']==gb['signature'];identical=ga['edges']==gb['edges']
            row=dict(model=modelid(k),panel=k[0],dataset=k[1],train_seed=k[2],arm=k[3],first=u,second=v,
                isomorphism='identical' if identical else ('excluded_by_joint_degree' if not same else 'unknown'),**overlap(ga['edges'],gb['edges']))
            local.append(row);pairrows.append(row)
        pairmodels.append(dict(model=modelid(k),panel=k[0],dataset=k[1],train_seed=k[2],arm=k[3],pairs=6,
            jaccard_mean=float(np.mean([r['jaccard'] for r in local])),jaccard_min=min(r['jaccard'] for r in local),jaccard_max=max(r['jaccard'] for r in local),
            certified_nonisomorphic=sum(r['isomorphism']=='excluded_by_joint_degree' for r in local),unknown=sum(r['isomorphism']=='unknown' for r in local)))
    save('within_model_pairs',pairrows);save('within_model_diversity',pairmodels)
    # Aggregation never treats individual generated graphs as training replications.
    buckets=defaultdict(list)
    for r in metric_rows:buckets[(r['model'],r['version'],r['metric'])].append(r['value'])
    modelrows=[];cross=defaultdict(list);modellookup={}
    for (mid,version,metric),vs in buckets.items():
        pa,ds,se,arm=mid.split('|');row=dict(model=mid,panel=pa,dataset=ds,train_seed=int(se),arm=arm,version=version,metric=metric,**moments(vs));modelrows.append(row)
        cross[(pa,ds,arm,version,metric)].append(row['mean'])
        if version=='original':modellookup.setdefault(mid,{})[metric]=row['mean']
    save('per_model',modelrows);save('across_training_seeds',[dict(panel=k[0],dataset=k[1],arm=k[2],version=k[3],metric=k[4],**moments(v)) for k,v in cross.items()])
    # All paired O->A and each output candidate versus its own input baseline.
    paircomparison=[]
    for k,items in pool.items():
        panel,ds,seed,arm=k
        baseline='O' if arm=='A' and panel=='matched_four_grid' else ('A' if arm=='PB' else arm.split('_')[0])
        if arm==baseline:continue
        bk=(panel,ds,seed,baseline)
        if bk not in pool:missing.append(dict(model=modelid(k),reason='missing paired baseline',baseline=str(bk)));continue
        base=dict(pool[bk])
        for uid,vals in items:
            buid=modelid(bk)+'|'+uid.split('|')[-1];bv=base[buid]
            d={m:(None if vals.get(m) is None or bv.get(m) is None else vals[m]-bv[m]) for m in CORE}
            paircomparison.append(dict(model=modelid(k),baseline=modelid(bk),dataset=ds,panel=panel,train_seed=seed,arm=arm,sample_seed=int(uid.split('|')[-1]),comparison=comparison(vals,bv),**d))
    save('paired_core_changes',paircomparison)
    # Complete within-dataset signature audit, no costly graph-isomorphism claims for ties.
    dsiso=[]
    for ds in ['IMDB','PubMed']:
        uu=[r['uid'] for r in records if r['dataset']==ds];different=identical=unknown=0
        for u,v in itertools.combinations(uu,2):
            if graphs[u]['edges']==graphs[v]['edges']:identical+=1
            elif graphs[u]['signature']!=graphs[v]['signature']:different+=1
            else:unknown+=1
        dsiso.append(dict(dataset=ds,graphs=len(uu),pairs=len(uu)*(len(uu)-1)//2,certified_nonisomorphic=different,identical=identical,unknown=unknown))
    save('isomorphism_inventory',dsiso);save('missing',missing)
    # Reference metrics are not thresholds: zero reference count uses null relative error.
    ranges=[]
    for k,items in pool.items():
        vals=[v for _,v in items];ref=reference_records[k[1]]
        row=dict(panel=k[0],dataset=k[1],train_seed=k[2],arm=k[3],samples=4)
        for metric in ['edges','triangles','lcc']:
            v=np.array([x['raw.'+metric] for x in vals]);rv=ref[metric];row.update({metric+'_reference':rv,metric+'_mean':float(v.mean()),metric+'_sample_SD':float(v.std(ddof=1)),metric+'_min':float(v.min()),metric+'_max':float(v.max()),metric+'_MAE':float(np.mean(abs(v-rv)))})
        row.update({m:float(np.mean([v[m] for v in vals])) for m in CORE if all(v[m] is not None for v in vals)})
        row['reference_jaccard_mean']=float(np.mean([r['jaccard'] for r in records if r['model']==modelid(k)]))
        ranges.append(row)
    save('model_overview',ranges)
    for path,h in HASHES.items():assert sha(path)==h,path
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets=HASHES))
    save('receipt',dict(status='complete_stopped',graphs=144,models=36,within_model_pairs=len(pairrows),original_metrics=144,
        lcc_metrics=sum(r['version']=='lcc' for r in full_index),seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_calls=0,training_updates=0,GPU_hours=0,new_graphs=0,thresholds_fitted=0,old_assets_unchanged=True,formal_training_authorized=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
