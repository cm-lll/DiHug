"""Read-only CPU audit of completed DE panel, no model forward or optimizer."""
import argparse
import ast
import datetime
import itertools
import json
import math
import pathlib
import time
from collections import Counter
import numpy as np
from independent_metrics import stats,errors,sha,close

def paths(types,edges,families,check):
    adj=[{} for _ in types]
    for (u,v),r in zip(edges.T,families):adj[u][v]=int(r);adj[v][u]=int(r)
    canon=lambda p:min(tuple(p),tuple(reversed(p)))
    out=[Counter(),Counter(),Counter()]
    for u in range(len(types)):
        check()
        for a,b in itertools.combinations(adj[u],2):out[1][canon([int(types[a]),adj[u][a],int(types[u]),adj[u][b],int(types[b])])]+=1
    for u,v in edges.T:
        check();r=adj[u][v];out[0][canon([int(types[u]),r,int(types[v])])]+=1
        for a in adj[u]:
            if a==v:continue
            for b in adj[v]:
                if b==u or b==a:continue
                out[2][canon([int(types[a]),adj[u][a],int(types[u]),r,int(types[v]),adj[v][b],int(types[b])])]+=1
    pairs=set();closed=set();numerator=denominator=0
    for m in np.flatnonzero(types==2):
        for d in adj[m]:
            if types[d]!=0:continue
            for a in adj[m]:
                if types[a]!=1:continue
                denominator+=1;pairs.add((d,a))
                if a in adj[d]:numerator+=1;closed.add((d,a))
    closure=dict(path_numerator=numerator,path_denominator=denominator,path_ratio=numerator/denominator if denominator else None,unique_pair_numerator=len(closed),unique_pair_denominator=len(pairs),unique_pair_ratio=len(closed)/len(pairs) if pairs else None)
    return [{','.join(map(str,k)):v for k,v in c.items()} for c in out],closure

def compare_value(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:close(a,b)

def tv(a,b):
    x,y=sum(a.values()),sum(b.values())
    return .5*sum(abs(a.get(k,0)/x-b.get(k,0)/y) for k in a.keys()|b.keys()) if x and y else 0. if not(x or y) else 1.

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--round1',required=True);p.add_argument('--round2',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.time();root=pathlib.Path(a.run);src=pathlib.Path(a.source);activation=json.loads((root/'activation.json').read_text())
    def check():
        if time.time()>=activation['deadline_epoch']-60:raise TimeoutError('original final deadline; stop audit')
    check();x=json.loads((root/'unified_evaluation.json').read_text());end=json.loads((root/'launcher_summary.json').read_text());cfg=json.loads((src/'package/config.json').read_text())
    assert x['status']==end['status']=='complete' and x['new_evaluated_graphs']==16 and x['old_evaluated_graphs']==32
    assert not end['primary_failures'] and len(end['jobs'])==8 and all(r['exit_code']==0 for r in end['jobs'])
    graphs=[];cells={};pairing=[];weights=[];hash_cache={}
    def hashed(p):
        key=str(p)
        if key not in hash_cache:hash_cache[key]=sha(p)
        return hash_cache[key]
    seeds={'IMDB':[2026102811,2026102834],'PubMed':[2026102812,2026102833]}
    import torch
    for ds,train_seeds in seeds.items():
        asset=src/'package'/cfg['datasets'][ds]['file'];assert hashed(asset)==cfg['datasets'][ds]['sha256']
        with np.load(asset,allow_pickle=False) as z:types=z['types'];schema=z['schema'];ref=stats(types,z['edges'].T,schema);ref_paths,_=paths(types,z['edges'].T,ref['fam'],check)
        for round_index,train_seed in enumerate(train_seeds):
            oldroot=pathlib.Path(a.round1 if round_index==0 else a.round2)
            oldinputs={arm:[json.loads(l)['input_sha256'] for l in (oldroot/(ds+'_'+arm)/'training.jsonl').read_text().splitlines()] for arm in 'ABC'}
            initial=[];rngs=[];input_sequences=[]
            for arm in 'DE':
                check();label=f'{ds}_s{train_seed}_{arm}';folder=root/label;receipt=json.loads((folder/'summary.json').read_text());binding=json.loads((folder/'binding.json').read_text());ini=json.loads((folder/'initialization.json').read_text())
                assert receipt['status']=='complete' and receipt['completed_updates']==1000 and receipt['completed_chains']==2
                assert binding['task_id']==label and binding['train_seed']==train_seed and binding['arm']==arm and binding['updates']==1000
                assert binding['source_sha256']==activation['source_sha256'] and all(ini['copied_exact'].values())
                ck=torch.load(folder/'initial.pt',map_location='cpu');assert ck['updates']==0 and ck['binding']==binding
                import hashlib
                actual={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in ck['model'].items()};assert actual==ini['tensor_sha256'];del ck
                initial.append(actual);rngs.append(json.loads((folder/'training_rng_reset.json').read_text()))
                oldinit=json.loads((oldroot/(ds+'_A')/'initialization.json').read_text())['tensor_sha256'];common=set(actual)&set(oldinit);assert all(actual[k]==oldinit[k] for k in common)
                rows=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()];assert [r['update'] for r in rows]==list(range(1,1001))
                assert all(r['optimizer_steps']==1 and len(r['inputs'])==2 and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in rows)
                inputs=[r['input_sha256'] for r in rows];assert all(inputs==v for v in oldinputs.values());input_sequences.append(inputs)
                assert rngs[-1]==json.loads((oldroot/(ds+'_A')/'training_rng_reset.json').read_text())
                for name,u in [('initial',0),('midpoint',500),('final',1000)]:
                    rec=json.loads((folder/(name+'.sha256.json')).read_text());assert rec['updates']==u and hashed(folder/(name+'.pt'))==rec['sha256']
                cells[label]=dict(updates=1000,parameters=ini['parameters'],normal2_total_seconds=sum(r['normal2_seconds'] for r in rows),postwarmup_mean_seconds=float(np.mean([r['normal2_seconds'] for r in rows if not r['warmup']])),peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib'],elapsed_since_activation=receipt['elapsed_since_activation'],sampling_seconds=[],validation_seconds=sum(r['seconds'] for r in json.loads((folder/'validation.json').read_text())))
                for seed in [2026102821,2026102822]:
                    check();s=folder/f'sample_{seed}';b=json.loads((s/'binding.json').read_text());assert b['binding']==binding and b['graph_sha256']==hashed(s/'graph.npz') and b['checkpoint_sha256']==hashed(folder/'final.pt') and b['metrics_sha256']==hashed(s/'metrics.json')
                    prior=json.loads((oldroot/(ds+'_A')/f'sample_{seed}/binding.json').read_text());assert b['initial_rng']==prior['initial_rng']
                    steps=[json.loads(l) for l in (s/'steps.jsonl').read_text().splitlines()];assert [r['time'] for r in steps]==list(range(1000,0,-1))
                    assert all(r['exact_coverage'] and r['blocks']==1 and r['covered_slots']==(13120 if ds=='IMDB' else 18336) for r in steps)
                    with np.load(s/'graph.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['types'],types);np.testing.assert_array_equal(z['schema'],schema);edges=z['edges'];m=stats(types,edges,schema);np.testing.assert_array_equal(m['fam'],z['families'])
                    old=x['cells'][label]['graphs'][str(seed)]['raw_metrics'];counts,closure=paths(types,edges,m['fam'],check)
                    for k in ['edges','triangles','lcc','isolates','maximum_degree','average_clustering']:close(m[k],old[k])
                    assert m['component_sizes']==old['component_sizes'] and dict(m['typed_triangles'])==old['typed_triangles']
                    np.testing.assert_array_equal(np.bincount(m['fam'],minlength=len(schema)),old['family_edges'])
                    assert m['mp']==[v['total'] for v in old['mp']] and counts==[v['counts'] for v in old['mp']]
                    for i in range(3):close(tv(counts[i],ref_paths[i]),old['comparison']['mp_frequency_TV'][i])
                    close(tv(dict(m['typed_triangles']),dict(ref['typed_triangles'])),old['primary']['typed_triangle_composition_TV'])
                    w,j=errors(types,schema,m['deg'],ref['deg']);close(w,old['W1_definitions']['incident4']);close(.4*w,old['W1_definitions']['all10_compat']);close(j,old['primary']['type_weighted_joint_degree_TV'])
                    if ds=='IMDB':
                        for k,v in closure.items():compare_value(v,old['imdb_closure'][k])
                        focus=x['cells'][label]['graphs'][str(seed)]['focus'];assert sorted(m['deg'][types==3].sum(1),reverse=True)==focus['Genre_sorted_degree_spectrum']['raw']
                    graphs.append(dict(cell=label,seed=seed,status='passed',W1_incident4=w,joint_degree_TV=j,steps=1000,graph_sha256=b['graph_sha256'],independent_MP_counts=True));cells[label]['sampling_seconds'].append(b['seconds'])
            assert input_sequences[0]==input_sequences[1] and rngs[0]==rngs[1]
            common=set(initial[0])&set(initial[1]);assert all(initial[0][k]==initial[1][k] for k in common)
            pairing.append(dict(dataset=ds,train_seed=train_seed,common_initial_tensor_hashes=True,actual_initial_bytes_verified=True,all_1000_normal2_inputs_equal_to_DE_and_historical_ABC=True,training_rng_equal_to_old_A=True,sampling_prior_rng_equal_to_old_A=True))
    # Independently recompute every stored contrast and two-sample mean, including null.
    count=0
    for c in x['contrasts']:
        d,s=c['dataset'],c['train_seed'];left,right=c['contrast'].split('-');differences=[]
        for row in c['pairs']:
            sample=str(row['sample_seed']);a=x['cells'][f'{d}_s{s}_{left}']['graphs'][sample]['values'];b=x['cells'][f'{d}_s{s}_{right}']['graphs'][sample]['values'];diff={}
            for target,source in [('raw_difference','raw'),('absolute_error_difference','absolute_error'),('distribution_error_difference','distribution_error')]:
                diff[target]={}
                for name,value in row['difference'][target].items():
                    av,bv=a[source].get(name),b[source].get(name);expected=None if av is None or bv is None else av-bv;compare_value(value,expected);diff[target][name]=expected;count+=1
            differences.append(diff)
        for group in differences[0]:
            for name,a in differences[0][group].items():
                b=differences[1][group][name];compare_value(c['mean'][group][name],None if a is None or b is None else (a+b)/2);count+=1
    for f in root.rglob('*.sha256.json'):
        check();ck=f.with_name(f.name.removesuffix('.sha256.json')+'.pt');assert hashed(ck)==json.loads(f.read_text())['sha256']
    payloads=[]
    for f in sorted(root.rglob('*')):
        if f.is_file():check();payloads.append(dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=hashed(f)))
    result=dict(status='passed',scope='CPU read-only independent discrete metrics, pair arithmetic, initialization/input/RNG receipts and bytes; no network forward, optimizer or new graph',graphs=graphs,cells=cells,pairing=pairing,pair_numeric_checks=count,payloads=payloads,seconds=time.time()-start,limitations=['two existing training seeds and two samples/model are exploratory','no extra GPU numerical test','labelled edge diversity is not an isomorphism-adjusted population claim'])
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='passed',graphs=len(graphs),cells=len(cells),pair_numeric_checks=count,seconds=result['seconds'])))
if __name__=='__main__':main()
