"""Independent raw-graph and every new-read audit; descriptive paired comparisons."""
from collections import defaultdict,Counter
import itertools,math,resource
from generation_common import *
sys.path.insert(0,str(EXP/'role_condition_coverage_v1'))
import coverage_evaluation as prior
prior.RESULT=OUT

def complete_features(row):
    f=c.features(row)
    for r,(a,b) in enumerate(itertools.combinations_with_replacement(range(4),2)):
        if r not in (1,2,5,8):continue
        for typ in set((a,b)):
            name=c.NAMES[typ];d=row['metrics']['typed_degree'][name]['by_relation'][r]
            for k in ('max','p90','p99','zero','low_le1'):f[f'{name}_relation{r}_{k}']=d[k]
    return f

def evaluate():
    tick=time.perf_counter();cpu=time.process_time();f=frozen();_,targets=c.load_data()
    rows=[read(p) for p in sorted(OUT.glob('*_202610201*/*_result.json'))]
    assert len(rows)==528
    expected=set(itertools.product(ARMS,SEEDS,f['targets'],f['sampling_seeds']))
    assert {(r['arm'],r['seed'],r['condition'],r['sampling_seed']) for r in rows}==expected
    groups=defaultdict(list);Gs={};random={};reads=Counter();maxima=defaultdict(float);times=defaultdict(list);first_difference=[]
    case=c.metadata_case(c.base,c.PANEL,targets);kernel=c.base.MarginalKernel(case['prior'],c.base.c.schedule(64))
    stepmap={};query_cache={}
    for i,r in enumerate(rows):
        check();folder=OUT/f"{r['arm']}_{r['seed']}";bind=read(folder/'binding.json')
        assert r['weight_sha256']==bind['weight']['sha256'] and r['newly_executed_chain']
        assert r['target_fingerprint']==targets[r['condition']].fingerprint
        for kind in ('graph','steps','trajectory','reads'):assert sha(OUT/r[kind+'_file'])==r[kind+'_sha256']
        G=prior.independent(r,targets);Gs[r['graph_file']]=G
        steps=read(OUT/r['steps_file']);rr=read(OUT/r['reads_file']);assert len(rr)==64 and [x['t'] for x in steps]==list(range(64,0,-1))
        assert all(x['exact_coverage'] and x['blocks']==4 and x['covered_slots']==312 for x in steps)
        rng=np.random.default_rng(r['sampling_seed']);initial=(rng.random(case['space'].total)<case['prior'].repeat(case['space'].counts)).astype(np.uint8)
        assert c.base.sha(initial.tolist())==r['initial_state_sha256']
        with np.load(OUT/r['trajectory_file']) as z:
            states=z['states'];assert states.shape==(64,case['space'].total)
            arrays={k:z[k] for k in z.files if k!='states'}
            before=initial
            for j,(s,recs) in enumerate(zip(steps,rr)):
                check();assert len(recs)==4 and [x['type'] for x in recs]==list(range(4)) and all(x['t']==s['t'] for x in recs)
                assert c.base.sha(states[j].tolist())==s['state_sha256'] and int(states[j].sum())==s['edge_count']
                assert np.all(states[j][case['active'].repeat(case['space'].counts)==0]==0)
                uniform=rng.random(case['space'].total);assert hashlib.sha256(uniform.tobytes()).hexdigest()==s['uniform_sha256']
                assert c.base.sha(rng.bit_generator.state)==s['rng_after']
                qkey=(r['sampling_seed'],s['t'])
                if qkey not in query_cache:
                    plan=c.base.c.plan_for(case,[r['sampling_seed'],s['t'],61],4);h=hashlib.sha256()
                    for b in range(4):
                        for ar in plan.block(b):h.update(np.ascontiguousarray(ar).tobytes())
                    query_cache[qkey]=h.hexdigest()
                assert query_cache[qkey]==s['query_sha256']
                if r['arm']=='Newton':
                    for typ,rec in enumerate(recs):
                        L,P,LP=[arrays[f'type{typ}_{k}'][j] for k in ('L','P','logP')]
                        counts=np.array([n for v,n in targets[r['condition']].roles[typ]])
                        independent=verify(P,LP,L,counts,rec)
                        assert rec['context_tail_bound']<=1e-250
                        reads[rec['status']]+=1
                        for key in ('row_residual','column_residual','stationarity_residual','total_omitted_mass_upper','context_tail_bound','iterations','function_evaluations'):
                            maxima[key]=max(maxima[key],rec[key])
                        cast=P.astype(np.float32).astype(float);vmax=rec['role_value_max'];k=P.shape[1];u=2**-24
                        maxima['FP32_context_conversion_bound']=max(maxima['FP32_context_conversion_bound'],float(abs(cast-P).sum(1).max()*vmax))
                        maxima['FP32_context_dot_bound']=max(maxima['FP32_context_dot_bound'],float(k*u/(1-k*u)*abs(cast).sum(1).max()*vmax+k*np.finfo(np.float32).tiny))
                        times['Newton_reads'].append(rec['total_checked_seconds'])
                else:
                    reads['exact_DP']+=4
                    times['DP_reads'] +=[x['total_checked_seconds'] for x in recs]
                before=states[j]
            edges,fam=c.base.c.state_edges(case,states[-1])
            with np.load(OUT/r['graph_file']) as graph:np.testing.assert_array_equal(edges.T,graph['edges']);np.testing.assert_array_equal(fam+1,graph['relations'])
        identity=(r['initial_state_sha256'],[(s['t'],s['uniform_sha256'],s['query_sha256'],s['rng_after']) for s in steps])
        key=r['sampling_seed']
        if key in random:assert random[key]==identity
        else:random[key]=identity
        stepmap[r['arm'],r['seed'],r['condition'],r['sampling_seed']]=steps
        groups[r['arm'],r['seed'],r['condition']].append(r);times[r['arm']+'_chains'].append(r['seconds'])
        if i%50==0:print('raw graph + all-read audit',i,'/528',flush=True)
    summaries=[];classes=[]
    import networkx as nx
    for (arm,seed,name),group in sorted(groups.items()):
        check();group.sort(key=lambda r:r['sampling_seed']);reps=[];freq=[];assign=[]
        for row in group:
            G=Gs[row['graph_file']];h=nx.weisfeiler_lehman_graph_hash(G,node_attr='type',edge_attr='relation');found=None
            for j,(other,oh) in enumerate(reps):
                if h==oh and nx.is_isomorphic(G,other,node_match=lambda x,y:x['type']==y['type'],edge_match=lambda x,y:x['relation']==y['relation']):found=j;break
            if found is None:found=len(reps);reps.append((G,h));freq.append(0)
            freq[found]+=1;assign.append(dict(sampling_seed=row['sampling_seed'],local_class=found))
        ff=[complete_features(r) for r in group];edge_diff=[]
        for a,b in itertools.combinations(group,2):
            edge_diff.append(len(set(Gs[a['graph_file']].edges())^set(Gs[b['graph_file']].edges())))
        target=sorted([v[8] for v,n in targets[name].roles[3] for _ in range(n)],reverse=True)
        summaries.append(dict(arm=arm,seed=seed,condition=name,category=prior.category(name,read(c.RESULT/'conditions.json')),graphs=8,
            statistics={k:c.stat([v[k] for v in ff]) for k in ff[0]},
            sorted_G_degree_mean=np.mean([r['target_errors']['Genre']['degrees_descending'] for r in group],axis=0).tolist(),target_G_degrees=target,
            topology_classes=len(freq),topology_frequencies=freq,mean_pair_edge_difference=float(np.mean(edge_diff)),
            effective_classes=math.exp(-sum(n/8*math.log(n/8) for n in freq))))
        classes.append(dict(arm=arm,seed=seed,condition=name,assignments=assign))
    differences=[]
    for seed,name in itertools.product(SEEDS,f['targets']):
        a=next(x for x in summaries if (x['arm'],x['seed'],x['condition'])==('DP',seed,name));b=next(x for x in summaries if (x['arm'],x['seed'],x['condition'])==('Newton',seed,name))
        differences.append(dict(seed=seed,condition=name,delta_Newton_minus_DP={k:b['statistics'][k]['mean']-a['statistics'][k]['mean'] if b['statistics'][k]['mean'] is not None and a['statistics'][k]['mean'] is not None else None for k in a['statistics']},topology_classes_delta=b['topology_classes']-a['topology_classes'],pair_edge_difference_delta=b['mean_pair_edge_difference']-a['mean_pair_edge_difference']))
        for ss in f['sampling_seeds']:
            aa=stepmap['DP',seed,name,ss];bb=stepmap['Newton',seed,name,ss];first=next((a['t'] for a,b in zip(aa,bb) if a['state_sha256']!=b['state_sha256']),None)
            first_difference.append(dict(seed=seed,condition=name,sampling_seed=ss,first_different_t=first))
    aggregate=[]
    for name in f['targets']:
        metrics={}
        for arm in ARMS:
            group=[x for x in summaries if x['arm']==arm and x['condition']==name]
            metrics[arm]={k:float(np.mean([x['statistics'][k]['mean'] for x in group])) if all(x['statistics'][k]['mean'] is not None for x in group) else None for k in group[0]['statistics']}
            metrics[arm]['mean_topology_classes']=float(np.mean([x['topology_classes'] for x in group]));metrics[arm]['mean_pair_edge_difference']=float(np.mean([x['mean_pair_edge_difference'] for x in group]))
        aggregate.append(dict(condition=name,**metrics))
    save(OUT/'generation.json',rows);save(OUT/'per_seed.json',dict(summaries=summaries,paired_differences=differences));save(OUT/'aggregate.json',aggregate)
    save(OUT/'topology_classes.json',classes);save(OUT/'trajectory_divergence.json',first_difference)
    receipts=[read(p) for p in OUT.glob('*_202610201*/receipt.json')];assert len(receipts)==6 and all(r['status']=='completed' and r['completed_chains']==88 for r in receipts)
    save(OUT/'audit.json',dict(status='passed',graphs=528,steps=33792,network_forwards=135168,read_counts=dict(reads),Newton_maxima=dict(maxima),
        timings={k:c.stat(v) for k,v in times.items()},receipts=receipts,wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        checks=['528 new complete chains','every Newton output independently residual/tail/KKT certified','all step states and final graph bound',
        'all paired prior/query/uniform/RNG hashes','independent degree/joint/W1/connectivity/typed triangles/full MP1/2/3/DMA closure','exact type+relation isomorphism within8draw cells'],
        not_checked=['network probabilities not independently recomputed','large graph network/GPU compatibility','population diversity from8draws'],training=0,GPU=0))
    print('AUDIT PASSED528',dict(reads),flush=True)
if __name__=='__main__':evaluate()
