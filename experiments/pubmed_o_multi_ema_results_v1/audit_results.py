"""Bounded CPU-only audit; reads frozen outputs, never constructs a model."""
import argparse,collections,datetime,hashlib,json,math,pathlib,sys,time
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def equal(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(a,b)
def tv(a,b):
    na,nb=sum(a.values()),sum(b.values())
    return .5*sum(abs(a.get(k,0)/na-b.get(k,0)/nb) for k in a.keys()|b.keys()) if na and nb else float(bool(na or nb))

def graph_stats(types,edges,schema):
    import networkx as nx
    assert edges.ndim==2 and edges.shape[0]==2 and np.all(edges[0]<edges[1])
    assert np.all(edges>=0) and np.all(edges<len(types)) and len(set(map(tuple,edges.T)))==edges.shape[1]
    lookup={tuple(p):i for i,p in enumerate(schema)}
    families=np.array([lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T])
    degree=np.zeros((len(types),len(schema)),dtype=np.int64)
    np.add.at(degree,(edges[0],families),1);np.add.at(degree,(edges[1],families),1)
    g=nx.Graph();g.add_nodes_from(range(len(types)));g.add_edges_from(edges.T)
    triangles=collections.Counter()
    for u,v in g.edges():
        for w in set(g[u])&set(g[v]):
            if u<v<w:triangles[str(tuple(sorted(map(int,types[[u,v,w]]))))]+=1
    components=sorted(map(len,nx.connected_components(g)),reverse=True)
    mp2=sum(d*(d-1)//2 for _,d in g.degree())
    mp3=sum((g.degree(u)-1)*(g.degree(v)-1)-len(set(g[u])&set(g[v])) for u,v in g.edges())
    return dict(families=families,degree=degree,edges=edges.shape[1],triangles=sum(triangles.values()),typed_triangles=triangles,
                component_sizes=components,lcc=components[0],isolates=len(list(nx.isolates(g))),maximum_degree=max(dict(g.degree()).values()),
                average_clustering=nx.average_clustering(g),mp=[edges.shape[1],mp2,mp3])

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    started=time.time();run=pathlib.Path(args.run);src=pathlib.Path(args.source);out=pathlib.Path(args.output)
    if out.exists():raise FileExistsError(out)
    activation=read(run/'activation.json')
    def check():
        if time.time()>=activation['deadline_epoch']-60:raise TimeoutError('original final deadline; audit stopped')
    sys.path.insert(0,str(src));import runtime_contract as c
    assert c.source_binding()[1]==activation['source_sha256']
    completed=read(run/'launcher_summary.json');evaluation=read(run/'unified_evaluation.json')
    assert completed['status']==evaluation['status']=='complete' and not completed['primary_failures'] and not completed['unstarted']
    assert len(completed['jobs'])==72 and all(j['exit_code']==0 for j in completed['jobs'])
    with np.load(src/'package/assets/PubMed.npz',allow_pickle=False) as z:
        types=z['types'];schema=z['schema'];ref=graph_stats(types,z['edges'].T,schema)
    import torch
    checkpoints=[];exports=[];graphs=[];cells=[];initial_rng_pairs={};hashes={};contrasts=0
    def hashed(p):
        check();key=str(p)
        if key not in hashes:hashes[key]=sha(p)
        return hashes[key]
    for seed in c.SEEDS:
        check();folder=run/c.cell(seed);binding=read(folder/'binding.json');receipt=read(folder/'summary.json')
        assert receipt['status']=='complete' and receipt['completed_updates']==receipt['ema_updates']==1000
        assert binding['train_seed']==seed and binding['arm']=='O' and binding['dataset']=='PubMed'
        rows=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,1001))
        assert all(r['optimizer_steps']==1 and r['ema_updates']==r['update'] and len(r['inputs'])==2 and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in rows)
        validation=read(folder/'validation.json');assert [v['updates'] for v in validation]==[0,500,1000]
        assert all(all(k in v for k in c.READOUTS) for v in validation)
        for name,u in [('initial',0),('midpoint',500),('final',1000)]:
            r=read(folder/(name+'.sha256.json'));assert r['updates']==u and hashed(folder/(name+'.pt'))==r['sha256']
            checkpoints.append(dict(train_seed=seed,updates=u,sha256=r['sha256'],path=str(folder/(name+'.pt'))))
        initial=torch.load(folder/'initial.pt',map_location='cpu');assert initial['updates']==initial['ema']['updates']==0
        for kind,decay in c.DECAYS.items():
            e=initial['ema']['members'][kind];assert e['decay']==decay and e['updates']==0
            assert all(torch.equal(initial['model'][k],v) for k,v in e['shadow'].items())
        initial_hashes={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in initial['model'].items()}
        if seed in c.NEW_SEEDS:
            proof=binding['initial']['construction'];assert all(proof['copied_exact'].values()) and initial_hashes==proof['tensor_sha256']
        else:
            sealed=torch.load(c.initial_path(seed),map_location='cpu');assert all(torch.equal(v,sealed['model'][k]) for k,v in initial['model'].items());del sealed
        del initial
        final=torch.load(folder/'final.pt',map_location='cpu');assert final['updates']==final['ema']['updates']==1000
        for kind in c.READOUTS:
            job=next(j for j in c.jobs()['sampling'] if j['train_seed']==seed and j['readout']==kind)
            path,readout_receipt=c.verify_readout(run,job);payload=torch.load(path,map_location='cpu')
            assert payload['train_seed']==seed and payload['readout']==kind and payload['updates']==1000 and payload['decay']==c.DECAYS.get(kind)
            expected=dict(final['model'])
            if kind!='raw':
                e=final['ema']['members'][kind];assert e['updates']==1000 and e['decay']==c.DECAYS[kind];expected.update(e['shadow'])
            assert payload['model'].keys()==expected.keys() and all(torch.equal(payload['model'][k],v) for k,v in expected.items())
            exports.append(dict(train_seed=seed,readout=kind,sha256=hashed(path),matches_final_exactly=True));del payload,expected
        del final
        cells.append(dict(train_seed=seed,updates=1000,training_total_update_seconds=sum(r['total_update_seconds'] for r in rows),normal2_component_seconds=sum(r['normal2_seconds'] for r in rows),validation_seconds=sum(v[k]['seconds'] for v in validation for k in c.READOUTS),peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib'],worker_seconds=receipt['finished_epoch']-receipt['started_epoch']))
    for job in c.jobs()['sampling']:
        check();folder=run/job['directory'];binding=read(folder/'binding.json');m=read(folder/'metrics.json')
        assert binding['job']==job and binding['source_sha256']==activation['source_sha256']
        assert hashed(folder/'graph.npz')==binding['graph_sha256'] and hashed(folder/'metrics.json')==binding['metrics_sha256']
        model,_=c.verify_readout(run,job);assert hashed(model)==binding['model_sha256']
        steps=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['time'] for r in steps]==list(range(1000,0,-1))
        assert all(r['exact_coverage'] and r['blocks']==1 and r['covered_slots']==18336 for r in steps)
        with np.load(folder/'graph.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['types'],types);np.testing.assert_array_equal(z['schema'],schema)
            s=graph_stats(types,z['edges'],schema);np.testing.assert_array_equal(s['families'],z['families'])
        for k in ['edges','triangles','lcc','isolates','maximum_degree','average_clustering']:equal(s[k],m[k])
        assert s['component_sizes']==m['component_sizes'] and dict(s['typed_triangles'])==m['typed_triangles']
        np.testing.assert_array_equal(np.bincount(s['families'],minlength=10),m['family_edges'])
        assert s['mp']==[v['total'] for v in m['mp']]
        w1=joint=0.
        for typ in np.unique(types):
            mask=types==typ;rels=[r for r,pair in enumerate(schema) if typ in pair];values=[]
            for r in rels:
                a,b=s['degree'][mask,r],ref['degree'][mask,r];n=int(max(a.max(),b.max()))+1
                values.append(np.abs(np.cumsum(np.bincount(a,minlength=n)/len(a)-np.bincount(b,minlength=n)/len(b))).sum())
            weight=float(mask.mean());w1+=weight*np.mean(values)
            a,b=collections.Counter(map(tuple,s['degree'][mask])),collections.Counter(map(tuple,ref['degree'][mask]));n=int(mask.sum())
            joint+=weight*.5*sum(abs(a[k]/n-b[k]/n) for k in a.keys()|b.keys())
        equal(w1,m['W1_definitions']['incident4']);equal(.4*w1,m['W1_definitions']['all10_compat']);equal(joint,m['primary']['type_weighted_joint_degree_TV'])
        equal(tv(s['typed_triangles'],ref['typed_triangles']),m['primary']['typed_triangle_composition_TV'])
        key=job['sample_seed'];pair=(binding['initial_state_sha256'],binding['initial_rng'])
        if key in initial_rng_pairs:assert pair==initial_rng_pairs[key]
        else:initial_rng_pairs[key]=pair
        graphs.append(dict(**job,status='passed',graph_sha256=binding['graph_sha256'],W1_incident4=w1,joint_degree_TV=joint,seconds=binding['seconds']))
    # Independently audit all saved numeric paired differences and null propagation.
    for contrast in evaluation['contrasts']:
        seed=contrast['train_seed'];kind=contrast['contrast'].removesuffix('-raw');diffs=[]
        for pair in contrast['per_sample']:
            sample=pair['sample_seed']
            find=lambda k:next(r for r in evaluation['cells'][c.cell(seed)+'_'+k]['graphs'] if r['sample_seed']==sample)['values']
            a,b=find(kind),find('raw');result={}
            for target,source in [('raw_difference','raw'),('absolute_error_difference','absolute_error'),('distribution_error_difference','distribution_error')]:
                result[target]={}
                for key,value in pair['difference'][target].items():
                    x,y=a[source].get(key),b[source].get(key);expected=None if x is None or y is None else x-y
                    equal(value,expected);result[target][key]=expected;contrasts+=1
            diffs.append(result)
        for group in diffs[0]:
            for key,a in diffs[0][group].items():
                b=diffs[1][group][key];equal(contrast['mean'][group][key],None if a is None or b is None else (a+b)/2);contrasts+=1
    for kind,decay in c.DECAYS.items():
        assessment=evaluation['assessment']['decays'][kind]
        for label,seeds in [('all_eight',c.SEEDS),('old_two',c.OLD_SEEDS),('new_six',c.NEW_SEEDS)]:
            differences=[]
            for seed in seeds:
                err=lambda k:np.mean([abs(r['metrics']['edges']-926) for r in evaluation['cells'][c.cell(seed)+'_'+k]['graphs']])
                d=err(kind)-err('raw');differences.append(d)
                pair=next(p for p in assessment['per_seed'] if p['train_seed']==seed);equal(pair['difference'],d)
            group=assessment['groups'][label];equal(group['mean_difference'],np.mean(differences));equal(group['median_difference'],np.median(differences))
            assert [group['improved'],group['tied'],group['worsened']]==[sum(d<0 for d in differences),sum(d==0 for d in differences),sum(d>0 for d in differences)]
    result=dict(status='passed',started=started,finished=time.time(),seconds=time.time()-started,source_sha256=activation['source_sha256'],graphs=graphs,cells=cells,checkpoints=checkpoints,readouts=exports,paired_numeric_checks=contrasts,actual_training_updates=8000,full_chain_count=64,scope='CPU read-only; all64 independent discrete graph/key metric checks;32 exact readout checks;24 checkpoint hashes; original two initial bytes and new6 construction receipts; no model forward',limits=['MP totals independently checked; MP composition retained from frozen evaluator, not independently re-enumerated','no new GPU test','eight seeds are one-data-set exploratory evidence; old2 participated in design'])
    check();out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['status','seconds','paired_numeric_checks','actual_training_updates','full_chain_count']}))
if __name__=='__main__':main()
