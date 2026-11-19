"""Cached real predictions -> CPU hard graph scoring; no model import/calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import sys,time,json,csv,hashlib,signal,resource,traceback,importlib.util,itertools
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/real_hard_feedback_cost_20261110'
OLD=ROOT/'results/hard_clean_pilot_v2_20261110'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
obj=module('score',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py')
loo=module('loo',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py').loo_gradient
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(name,z): (OUT/name).write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False))
def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def inputhash(x):
    h=hashlib.sha256();h.update(str(x['t']).encode())
    for k in ('ids','query','f','edges','edge_families','labels'):h.update(np.ascontiguousarray(x[k]).tobytes())
    return h.hexdigest()
def independent_features(core,bits):
    # Node-centered neighbor-pair enumeration, distinct from scoring edge/common-neighbor loop.
    adj=[set() for _ in core.types];degree=np.zeros_like(core.ref['degree']);cnt=np.zeros(core.r,dtype=int)
    for (u,v),f in zip(core.q[:,bits.astype(bool)].T,core.f[bits.astype(bool)]):
        adj[u].add(int(v));adj[v].add(int(u));degree[u,f]+=1;degree[v,f]+=1;cnt[f]+=1
    tri=np.zeros(len(core.triples),dtype=int)
    for u,nb in enumerate(adj):
        for v,w in itertools.combinations(sorted(x for x in nb if x>u),2):
            if w in adj[v]:tri[core.tindex[tuple(sorted(core.types[[u,v,w]]))]]+=1
    unseen=set(range(core.n));sizes=[]
    while unseen:
        todo=[unseen.pop()];n=0
        while todo:
            u=todo.pop();n+=1;new=adj[u]&unseen;unseen-=new;todo.extend(new)
        sizes.append(n)
    return dict(counts=cnt,degree=degree,triangles=tri,components=len(sizes),lcc=max(sizes),
        isolated=sum(not x for x in adj),max_degree=max(map(len,adj)))
def main():
    start=time.time();OUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('540 seconds CPU worker bound')));signal.alarm(540)
    rec=dict(start_epoch=start,status='running',GPU_hours=0,model_calls=0,updates=0,reverse_chains=0,new_hard_graphs=0,old_score_checks=0,independent_feature_checks=0)
    batches=[];groups=[];graphs=[];pairs=[];assets=[];cases=[];cache={};bindings={}
    def bind(p):
        p=Path(p);v=sha(p);bindings[str(p)]=v;return v
    try:
        for p in (Path(__file__),Path(__file__).with_name('watch.py'),ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py',ROOT/'research_branches/hard_clean_pilot_v2_20261110/worker.py',OLD/'bindings.json'):
            if p.exists():bind(p)
        protocol=dict(track='EVALUATION supporting CORE hard clean structure objective',datasets={'IMDB':[2026110411,2026110412],'PubMed':[2026110421,2026110422]},updates=2000,readout='raw before hard-objective updates',times=[1,75,250,550,800,1000],K=64,repeats=4,hard_stream=2026111057,
            score='unchanged HardScore five groups; no edge count constraint, no weight adjustment',gradient='LOO derivative with respect to binary logit margin only; no network parameter Jacobian or projection',
            independent_check='old 32 draws/case all rescored; first and last new draws per repeat independently count triangles/degree/components; direct leave-one-out estimator on all new draws',
            limits={'threads':1,'RSS_GiB':2,'worker_seconds':540,'watch_seconds':600,'GPU':0},
            limitations=['fixed forward-corrupted inputs from same reference, one per time/model','four Monte Carlo repeats not independent training seeds','no exact real graph gradient available; pairwise consistency not unbiasedness test','legacy scoring scales retained including zero-reference triangle squared counts'])
        dump('protocol.json',protocol);dump('freeze.json',dict(epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json')))
        # Bind and validate the entire panel before inspecting any new candidate scores.
        for ds,seeds in protocol['datasets'].items():
            for seed in seeds:
                folder=OLD/f'{ds}_{seed}_HARD_MGDA';identity=json.loads((folder/'identity.json').read_text());row=identity['row']
                for n in ('identity.json','evaluation_inputs.pt','evaluation.json','score_definition.json','environment.json'):bind(folder/n)
                assert bind(row['data'])==row['data_sha256'];assert bind(row['checkpoint'])==row['checkpoint_sha256']
                z=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
                assert z['updates']==z['ema']['updates']==identity['parent_updates']==2000
                assert z['binding']['dataset']==ds and z['binding']['train_seed']==seed and z['binding']['arm']=='A'
                cfg=z['binding']['config'];assert cfg['inputs_per_update']==cfg['rounds']==1
                del z
                d=np.load(row['data']);types=d['types'];edges={tuple(sorted(e)) for e in d['edges']};schema=d['schema'];assert len(types)==192 and len(edges)==(323 if ds=='IMDB' else 926)
                v=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False);ev=json.loads((folder/'evaluation.json').read_text());assert [x['t'] for x in v['inputs']]==protocol['times']
                for inp,ih in zip(v['inputs'],v['hashes']):
                    assert inputhash(inp)==ih;t=int(inp['t']);path=folder/f'pred_0_raw_{t}.npz';bind(path);pred=np.load(path)
                    order=np.argsort(inp['ids']);q=np.asarray(inp['query'])[:,order];f=np.asarray(inp['f'])[order];ref=np.asarray(inp['labels'])[order]
                    np.testing.assert_array_equal(pred['ids'],np.asarray(inp['ids'])[order]);np.testing.assert_array_equal(ref,[int(tuple(e) in edges) for e in q.T])
                    np.testing.assert_array_equal(np.sort(types[q].T,axis=1),np.sort(schema[f],axis=1));assert int(ref.sum())==len(edges)
                    e=next(e for e in ev if e['offset']==0 and e['readout']=='raw' and e['t']==t);assert e['updates']==2000 and e['input_sha256']==ih
                    p=pred['probability'].astype(np.float64);assert p.shape==ref.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    if ds not in cache:
                        core=obj.HardScore(types,q,f,ref,len(schema));cache[ds]=core
                        np.testing.assert_array_equal(core.costs(ref)[0],np.zeros(5))
                        sd=json.loads((folder/'score_definition.json').read_text())
                        for k,val in core.ref.items():np.testing.assert_array_equal(val,sd['reference'][k])
                    core=cache[ds]
                    for a,b in ((core.q,q),(core.f,f),(core.reference,ref)):np.testing.assert_array_equal(a,b)
                    cases.append((ds,seed,t,p,path,ih));assets.append(dict(dataset=ds,seed=seed,t=t,updates=2000,readout='raw',checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],data=row['data'],input_hash=ih,prediction=str(path),prediction_sha256=sha(path),slots=len(p),fixed_zero_slots=18336-len(p)))
                print('bound',ds,seed,flush=True)
        dump('assets.json',assets);dump('bindings.json',bindings)
        maxloo=0.;maxold=0.
        for ds,seed,t,p,path,ih in cases:
            core=cache[ds];tag=f'{ds}_{seed}_t{t}';pred=np.load(path)
            old=core.batch(pred['bits']);np.testing.assert_allclose(old,pred['costs'],atol=1e-12,rtol=1e-12);maxold=max(maxold,float(abs(old-pred['costs']).max()));rec['old_score_checks']+=len(old)
            allbits=[];allcost=[];allgrad=[]
            for rep in range(4):
                meta=dict(dataset=ds,seed=seed,t=t,repeat=rep);rng=np.random.default_rng(np.random.SeedSequence([2026111057,seed,t,rep]));s=time.perf_counter();bits=(rng.random((64,len(p)))<p).astype(np.uint8);sampling=time.perf_counter()-s
                s=time.perf_counter();vals=[core.costs(b) for b in bits];cost=np.stack([v[0] for v in vals]);scoring=time.perf_counter()-s
                s=time.perf_counter();g=loo(p,bits,cost);gradient=time.perf_counter()-s;assert np.isfinite(g).all()
                # Independent literal mean-of-other-samples formulation.
                direct=sum(np.outer(cost[k]-np.delete(cost,k,axis=0).mean(0),bits[k]-p) for k in range(64))/64
                np.testing.assert_allclose(g,direct,atol=1e-12,rtol=1e-9);maxloo=max(maxloo,float(abs(g-direct).max()))
                for k in (0,63):
                    alt=independent_features(core,bits[k])
                    for name in alt:np.testing.assert_array_equal(alt[name],vals[k][1][name])
                    rec['independent_feature_checks']+=1
                norms=np.linalg.norm(g,axis=1);total=float(norms.sum());common=np.zeros_like(g)
                for f in np.unique(core.f):common[:,core.f==f]=g[:,core.f==f].mean(1)[:,None]
                batches.append(dict(**meta,sampling_seconds=sampling,scoring_seconds=scoring,gradient_seconds=gradient,score_and_feedback_seconds=sampling+scoring+gradient,unique_hard_graphs=len(np.unique(np.packbits(bits,axis=1),axis=0)),zero_gradient_groups=int((norms==0).sum()),mean_gradient_norm=float(np.linalg.norm(g.mean(0))),min_probability=float(p.min()),max_probability=float(p.max()),exact_zero_probability=int((p==0).sum()),exact_one_probability=int((p==1).sum())))
                for j,name in enumerate(core.names):
                    groups.append(dict(**meta,group=name,cost_mean=float(cost[:,j].mean()),cost_std=float(cost[:,j].std(ddof=1)),cost_min=float(cost[:,j].min()),cost_max=float(cost[:,j].max()),zero_cost_samples=int((cost[:,j]==0).sum()),constant_cost=bool(np.ptp(cost[:,j])==0),gradient_norm=float(norms[j]),norm_share=float(norms[j]/total) if total else 0.,relation_common_squared_fraction=float(np.sum(common[j]**2)/norms[j]**2) if norms[j] else 0.,positive_gradient_mean=float(g[j,core.reference==1].mean()),nonedge_gradient_mean=float(g[j,core.reference==0].mean())))
                for k,(c,v) in enumerate(vals):
                    graphs.append(dict(**meta,sample=k,edges=int(v['counts'].sum()),triangles=int(v['triangles'].sum()),component_count=v['components'],lcc=v['lcc'],isolated=v['isolated'],max_degree=v['max_degree'],**dict(zip(core.names,map(float,c)))))
                allbits.append(bits);allcost.append(cost);allgrad.append(g);rec['new_hard_graphs']+=64
            G=np.stack(allgrad)
            for i,j in itertools.combinations(range(4),2):
                for k,name in enumerate(core.names+['arithmetic_mean']):
                    a,b=(G[i,k],G[j,k]) if k<5 else (G[i].mean(0),G[j].mean(0));n=float(np.linalg.norm(a)*np.linalg.norm(b))
                    pairs.append(dict(dataset=ds,seed=seed,t=t,repeat_a=i,repeat_b=j,group=name,cosine=float(a@b/n) if n else None,both_nonzero=bool(n)))
            np.savez_compressed(OUT/f'{tag}.npz',probability=p,ids=pred['ids'],bits=np.stack(allbits),costs=np.stack(allcost),logit_gradients=G)
            for name,rows in [('batches.csv',batches),('groups.csv',groups),('graphs.csv',graphs),('repeat_pairs.csv',pairs)]:csvwrite(name,rows)
            dump('receipt.json',dict(rec,elapsed_seconds=time.time()-start));print('completed',tag,'seconds',round(time.time()-start,2),flush=True)
        for p0,h in bindings.items():assert sha(p0)==h,'source changed '+p0
        dump('audit.json',dict(passed=True,max_old_score_error=maxold,max_LOO_error=maxloo,reference_zero_cost=True,all_asset_hashes_unchanged=True,independent_feature_checks=rec['independent_feature_checks']))
        rec['status']='completed'
    except BaseException as e:
        rec['status']='failed';rec['error']=repr(e);(OUT/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        rec.update(elapsed_seconds=time.time()-start,end_epoch=time.time(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump('receipt.json',rec);signal.alarm(0)
if __name__=='__main__':main()
