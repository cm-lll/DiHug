"""CPU, frozen one-candidate heterogeneous heat-signature feasibility audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json, csv, hashlib, time, resource, traceback, importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, wasserstein_distance

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/typed_heat_audit_20261111'
TIMES=np.logspace(-1,2,9)
spec=importlib.util.spec_from_file_location('old_score',ROOT/'research_branches/log_risk_cpu_20261111/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def table(name,rows):
    save(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def heat(a,types):
    start=time.perf_counter();n=len(a);levels=np.unique(types);d=a.sum(1)
    inv=np.divide(1,np.sqrt(d),out=np.zeros(n),where=d>0)
    lap=np.diag((d>0).astype(float))-inv[:,None]*a*inv[None,:]
    eig,v=eigh(lap,check_finite=True,driver='evr')
    assert eig.min()>-1e-10 and eig.max()<2+1e-10
    h=np.exp(-TIMES[:,None]*eig)
    ret=(v*v)@h.T
    indicator=np.column_stack([types==t for t in levels]).astype(float)
    projected=v.T@(np.sqrt(d)[:,None]*indicator)
    reach=np.stack([inv[:,None]*(v@(h_s[:,None]*projected)) for h_s in h],axis=1)
    reach[d==0]=indicator[d==0,None,:]
    assert np.max(abs(reach.sum(2)-1))<1e-9
    assert reach.min()>-1e-9 and ret.min()>-1e-9
    # Equal weight to return and type-arrival blocks, not each type channel.
    feat=np.concatenate([ret/np.sqrt(2*len(TIMES)),reach.reshape(n,-1)/np.sqrt(2*len(TIMES)*len(levels))],1)
    return dict(global_trace=h.mean(1),ret=ret,reach=reach,feat=feat,
                seconds=time.perf_counter()-start,eigenvalues=eig)

def bandwidths(ref,types):
    out={}
    for t in np.unique(types):
        ds=pdist(ref['feat'][types==t],metric='sqeuclidean');ds=ds[ds>1e-24]
        out[int(t)]=float(np.median(ds)) if len(ds) else 1.0
    return out

def score(h,ref,types,sigmas):
    terms=[]
    for t in np.unique(types):
        x=h['feat'][types==t];y=ref['feat'][types==t];s=sigmas[int(t)]
        kernel=lambda a,b:np.exp(-cdist(a,b,'sqeuclidean')/(2*s))
        value=float((kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean())/2)
        assert value>-1e-10
        terms.append(value)
    return float(np.mean((h['global_trace']-ref['global_trace'])**2)),float(np.mean(terms)),terms

def adj(n,edges):
    a=np.zeros((n,n));edges=np.asarray(edges,dtype=int).reshape(-1,2)
    if len(edges):a[edges[:,0],edges[:,1]]=1;a[edges[:,1],edges[:,0]]=1
    assert np.array_equal(a,a.T) and not np.diag(a).any()
    return a

def checks():
    types=np.array([0,0,0,1,1,1]);a=adj(6,[(i,i+1) for i in range(5)])
    h=heat(a,types);d=a.sum(1);inv=1/np.sqrt(d);lap=np.eye(6)-inv[:,None]*a*inv[None,:]
    error=0
    for j,s in enumerate(TIMES):
        e=expm(-s*lap);rw=inv[:,None]*e*np.sqrt(d)[None,:]
        error=max(error,float(np.max(abs(np.diag(e)-h['ret'][:,j]))))
        np.testing.assert_allclose(rw@np.column_stack([types==t for t in np.unique(types)]),h['reach'][:,j],atol=1e-10,rtol=1e-9)
    assert error<1e-10
    sig=bandwidths(h,types);assert abs(score(h,h,types,sig)[1])<1e-12
    permutation=np.array([4,2,0,5,1,3]);hp=heat(a[np.ix_(permutation,permutation)],types[permutation])
    np.testing.assert_allclose(hp['feat'],h['feat'][permutation],atol=1e-10)
    # Same untyped graph, changed type assignment: global trace unchanged.
    other_types=np.array([0,1,0,1,0,1]);changed=heat(a,other_types)
    typed_terms=[]
    for t in [0,1]:
        x=changed['feat'][other_types==t];y=h['feat'][types==t];s=sig[t]
        ker=lambda x,y: np.exp(-cdist(x,y,'sqeuclidean')/(2*s))
        typed_terms.append(float((ker(x,x).mean()+ker(y,y).mean()-2*ker(x,y).mean())/2))
    assert np.array_equal(h['global_trace'],changed['global_trace']) and np.mean(typed_terms)>1e-8
    empty=heat(np.zeros((6,6)),types);np.testing.assert_allclose(empty['ret'],1,atol=1e-12)
    c6=adj(6,[(i,(i+1)%6) for i in range(6)])
    two3=adj(6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])
    cyc_gap=float(np.linalg.norm(heat(c6,types)['global_trace']-heat(two3,types)['global_trace']))
    assert cyc_gap>1e-8
    return dict(expm_max_error=error,permutation_pass=True,type_relabel_typed_cost=float(np.mean(typed_terms)),
                type_relabel_global_gap=0,empty_pass=True,c6_two_c3_trace_gap=cyc_gap,
                meaning='formula fixtures, not generation-quality evidence')

def mutations(core,seed):
    rng=np.random.default_rng(seed);bits=core.reference.copy();active=np.flatnonzero(bits)
    deleted=bits.copy();deleted[rng.choice(active,max(1,len(active)//4),replace=False)]=0
    available=np.flatnonzero((bits==0)&np.isin(core.f,np.unique(core.f[active])))
    added=bits.copy();added[rng.choice(available,min(len(available),max(1,len(active)//4)),replace=False)]=1
    swapped=bits.copy();lookup={tuple(e):i for i,e in enumerate(core.q.T)};accepted=0;target=max(1,len(active)//4)
    for attempt in range(100*target):
        if accepted==target:break
        f=int(rng.choice(np.unique(core.f[active])));idx=np.flatnonzero((swapped==1)&(core.f==f))
        if len(idx)<2:continue
        e1,e2=rng.choice(idx,2,replace=False);u,v=map(int,core.q[:,e1]);w,x=map(int,core.q[:,e2])
        if core.types[u]>core.types[v]:u,v=v,u
        if core.types[w]>core.types[x]:w,x=x,w
        if core.types[u]==core.types[v] and rng.random()<.5:w,x=x,w
        if len({u,v,w,x})<4:continue
        p=lookup.get(tuple(sorted((u,x))));q=lookup.get(tuple(sorted((w,v))))
        if p is None or q is None or swapped[p] or swapped[q] or core.f[p]!=f or core.f[q]!=f:continue
        swapped[[e1,e2]]=0;swapped[[p,q]]=1;accepted+=1
    np.testing.assert_array_equal(core.features(swapped)['degree'],core.features(bits)['degree'])
    return [('delete_25pct',deleted,0),('add_25pct',added,0),('relation_degree_preserving_swaps',swapped,accepted)]

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/log_risk_cpu_20261111/run2'
    oldrows=json.loads((source/'existing_graph_scores.json').read_text())
    graph_hashes=json.loads((source/'graph_bindings.json').read_text())
    tasks=json.loads((ROOT/'results/real192_update_audit_20261111/protocol.json').read_text())['tasks']
    bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(old.__file__)):sha(old.__file__),str(Path(old.lib.__file__)):sha(old.lib.__file__)}
    bindings.update(graph_hashes)
    for t in tasks:
        r=t['parent'];assert sha(r['data'])==r['data_sha256'];bindings[r['data']]=r['data_sha256']
        p=ROOT/'results/real192_update_audit_20261111'/t['id']/'validation_u2000_raw_t1.npz';bindings[str(p)]=sha(p)
    for p,v in bindings.items():assert sha(p)==v
    save('protocol.json',dict(frozen_before_scoring=True,times=TIMES.tolist(),
        candidate='Normalized-Laplacian node return + random-walk type arrival joint distribution, per-type Gaussian V-statistic MMD2/2; equal blocks/types',
        isolated='L diagonal zero; random-walk heat leaves isolated vertex at itself',
        baseline='untyped normalized heat trace / n; mean squared signature gap',
        bandwidth='per-type median positive squared reference feature distance; >1e-24; fallback 1; frozen from reference only',
        scope='48 old whole graphs, two references and empty controls, 18 CPU perturbations (three fixed seeds x three transformations x two data)',
        seeds=[2026111161,2026111162,2026111163],
        limit='single CPU, 600 seconds; no GPU/model/update/reverse generation',
        no_claims=['Not original NetLSD replication','No complete ordered metapath encoding','No training or unbiased gradient validation','No test-set selection'],bindings=bindings))
    save('checks.json',checks());cores={};refs={};sigmas={};rows=[];signatures={};hash_status=[]
    for task in tasks:
        dataset=task['parent']['dataset']
        if dataset in cores:continue
        data=np.load(task['parent']['data']);z=np.load(ROOT/'results/real192_update_audit_20261111'/task['id']/'validation_u2000_raw_t1.npz')
        schema=data['schema'];assert len(set(map(tuple,schema)))==len(schema)
        core=old.FastScore(data['types'],z['query'],z['families'],z['labels'],len(schema));cores[dataset]=core
        for f,(u,v) in enumerate(schema):
            slots=core.q[:,core.f==f]
            assert np.all(np.sort(core.types[slots],axis=0)==np.array(sorted((u,v)))[:,None])
        ar=adj(core.n,core.q[:,core.reference.astype(bool)].T)
        np.testing.assert_array_equal(ar,adj(core.n,data['edges']))
        refs[dataset]=heat(ar,core.types);sigmas[dataset]=bandwidths(refs[dataset],core.types)
        hash_status.append(dict(dataset=dataset,n=core.n,edges=int(core.reference.sum()),types=len(np.unique(core.types)),schema=schema.tolist(),
            unique_relation_per_endpoint_types=True,type_sizes={int(t):int((core.types==t).sum()) for t in np.unique(core.types)},bandwidth_squared=sigmas[dataset]))
    save('assets.json',hash_status)
    def evaluate(dataset,task,arm,path,bits,seed=None,accepted=None):
        if time.time()-start>600:raise TimeoutError('CPU budget reached')
        core=cores[dataset];a=adj(core.n,core.q[:,bits.astype(bool)].T)
        h=heat(a,core.types);tick=time.perf_counter();g,c,terms=score(h,refs[dataset],core.types,sigmas[dataset]);dist_time=time.perf_counter()-tick
        tick=time.perf_counter();cost,feat=core.costs(bits);old_time=time.perf_counter()-tick
        deg=feat['degree'].sum(1);refdeg=core.features(core.reference)['degree'].sum(1)
        row=dict(dataset=dataset,task=task,arm=arm,path=path,seed=seed,accepted_swaps=accepted,
            edges=int(bits.sum()),triangles=int(feat['triangles'].sum()),components=int(feat['components']),lcc=int(feat['lcc']),
            isolated=int(feat['isolated']),max_degree=int(deg.max()),degree_W1=float(wasserstein_distance(deg,refdeg)),
            global_heat_cost=g,typed_heat_cost=c,typed_group_costs=json.dumps(terms),
            old_score=float(cost.mean()),old_log_score=float(np.log1p(cost).mean()),
            relation_count_cost=float(cost[0]),joint_degree_cost=float(cost[1]),present_triangle_cost=float(cost[2]),absent_triangle_cost=float(cost[3]),components_cost=float(cost[4]),
            heat_seconds=h['seconds'],distance_seconds=dist_time,old_cost_seconds=old_time)
        key=f'graph_{len(rows):03d}';signatures[key+'_global']=h['global_trace'];signatures[key+'_features']=h['feat'];signatures[key+'_bits']=bits
        row['signature_key']=key;rows.append(row)
    for dataset,core in cores.items():
        evaluate(dataset,'control','reference',None,core.reference)
        evaluate(dataset,'control','empty_control',None,np.zeros_like(core.reference))
    for row in oldrows:
        if row['path'] is None:continue
        dataset=row['task'].split('_')[0];core=cores[dataset];z=np.load(row['path']);np.testing.assert_array_equal(z['types'],core.types)
        edges=z['edges'];edges=edges.T if edges.shape[0]==2 else edges
        lookup={tuple(e):i for i,e in enumerate(core.q.T)};bits=np.zeros(len(core.reference),dtype=np.uint8)
        for u,v in edges:bits[lookup[tuple(sorted((int(u),int(v))))]]=1
        assert bits.sum()==len(edges)
        evaluate(dataset,row['task'],row['arm'],row['path'],bits)
    for dataset,core in cores.items():
        for seed in [2026111161,2026111162,2026111163]:
            for arm,bits,count in mutations(core,seed):evaluate(dataset,'cpu_perturbation',arm,None,bits,seed,count)
    table('per_graph',rows);np.savez_compressed(OUT/'signatures.npz',**signatures)
    summaries=[]
    for task in sorted(set(r['task'] for r in rows)):
        for dataset in cores:
            for arm in sorted(set(r['arm'] for r in rows)):
                group=[r for r in rows if r['task']==task and r['dataset']==dataset and r['arm']==arm]
                if not group:continue
                rec=dict(task=task,dataset=dataset,arm=arm,n=len(group))
                for k in ['global_heat_cost','typed_heat_cost','old_log_score','edges','triangles','degree_W1','joint_degree_cost','components']:
                    rec[k+'_mean']=float(np.mean([r[k] for r in group]));rec[k+'_sd']=float(np.std([r[k] for r in group],ddof=1)) if len(group)>1 else None
                summaries.append(rec)
    table('summary',summaries)
    correlations=[]
    for dataset in cores:
        for panel in ['all_old_graphs','ordinary_CE_only']:
            rr=[r for r in rows if r['dataset']==dataset and r['path'] and (panel=='all_old_graphs' or r['arm']=='ordinary_CE')]
            for metric in ['degree_W1','joint_degree_cost','present_triangle_cost','absent_triangle_cost','components_cost','relation_count_cost']:
                for scorer in ['global_heat_cost','typed_heat_cost']:
                    x=[r[scorer] for r in rr];y=[r[metric] for r in rr]
                    rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
                    correlations.append(dict(dataset=dataset,panel=panel,n=len(rr),scorer=scorer,metric=metric,spearman=rho))
    table('correlations',correlations)
    for p,v in bindings.items():assert sha(p)==v
    save('receipt.json',dict(status='complete',seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        threads=1,existing_graphs=48,cpu_perturbed_graphs=18,reference_controls=4,rows=len(rows),GPU_hours=0,model_calls=0,training_updates=0,reverse_chains=0,
        median_heat_seconds=float(np.median([r['heat_seconds'] for r in rows])),median_distance_seconds=float(np.median([r['distance_seconds'] for r in rows])),
        median_old_score_seconds=float(np.median([r['old_cost_seconds'] for r in rows])),source_assets_unchanged=True))
    print(json.dumps({'receipt':json.loads((OUT/'receipt.json').read_text()),'summary':summaries},indent=2))

if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save('failure.json',{'traceback':traceback.format_exc()})
        raise
