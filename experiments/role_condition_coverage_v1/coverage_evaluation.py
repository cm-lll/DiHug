"""All528 actual graphs, independent integer/path audit and paired seed summaries.

Unique module name prevents collision with historical evaluate.py adapters.
"""
from collections import defaultdict,Counter
import itertools,math
import networkx as nx
from common import *

def independent(row,targets):
    G,d,counts,types=independent_graph(RESULT/row['graph_file']);m=row['metrics'];targ=targets[row['condition']].as_dict()
    assert counts.tolist()==m['family_edges'] and G.number_of_edges()==m['edges']
    assert len(G)==m['nodes']==34
    cc=sorted(map(len,nx.connected_components(G)),reverse=True)
    assert cc==m['component_sizes'] and m['lcc']==cc[0] and m['isolates']==nx.number_of_isolates(G)
    assert m['connected']==(len(cc)==1)
    for typ,name in enumerate(NAMES):
        rows=d[types==typ]
        assert dict(Counter(','.join(map(str,v)) for v in rows))==m['joint_relation_degree'][name]
        for r in range(10):assert np.bincount(rows[:,r]).tolist()==m['typed_degree'][name]['by_relation'][r]['histogram']
    hist=sorted(d[types==3,8]);want=sorted(v[8] for v,n in targ['roles'][3] for _ in range(n))
    assert abs(np.mean(abs(np.array(hist)-want))-row['target_errors']['Genre']['W1'])<1e-12
    assert row['target_errors']==error_stats(m,targ)
    triangles=Counter()
    for tri in itertools.combinations(G,3):
        if all(G.has_edge(u,v) for u,v in itertools.combinations(tri,2)):triangles[tuple(sorted(int(types[x]) for x in tri))]+=1
    assert sum(triangles.values())==m['triangles']
    assert {str(k):v for k,v in sorted(triangles.items())}==m['typed_triangles']
    # Independent node-simple DFS; count reverse once by numeric endpoint orientation.
    paths=[Counter(),Counter(),Counter()]
    def visit(nodes,pattern):
        if len(nodes)>1 and nodes[0]<nodes[-1]:
            key=min(tuple(pattern),tuple(reversed(pattern)));paths[len(nodes)-2][','.join(map(str,key))]+=1
        if len(nodes)==4:return
        for v,edge in G[nodes[-1]].items():
            if v in nodes:continue
            visit(nodes+[v],pattern+[edge['relation'],int(types[v])])
    for u in G:visit([u],[int(types[u])])
    for a,b in zip(paths,m['mp']):assert dict(a)==b['counts'] and sum(a.values())==b['total']
    closure=[(u,mov,v) for mov in np.flatnonzero(types==2) for u in G[mov] for v in G[mov] if types[u]==0 and types[v]==1]
    closed=[x for x in closure if G.has_edge(x[0],x[2])];c=m['imdb_closure']
    assert [len(closed),len(closure),len({(u,v) for u,_,v in closed}),len({(u,v) for u,_,v in closure})]==[
        c['path_numerator'],c['path_denominator'],c['unique_pair_numerator'],c['unique_pair_denominator']]
    return G

def category(name,conditions):
    if name in ('balanced','extreme'):return 'original_controls'
    if name.startswith('dev_'):return 'seen_development'
    if name==conditions['positive_control']:return 'training_middle_control'
    if name in conditions['combination_tests']:return 'unseen_combinations'
    return 'unseen_values'

def audit():
    start=time.monotonic();cfg=settings();records,targets=load_data();conditions=read(RESULT/'conditions.json')
    # All data source roles and all train-validation isomorphism exclusions rechecked.
    train=[r for r in records if r['split']=='train'];vals=[r for r in records if r['split']=='validation']
    for arm in cfg['arms']:assert sum(r['arm']==arm for r in train)==48
    for r in records:assert base.extract_target(r['case'])==targets[r['condition']]
    for v in vals:
        check()
        for r in train:
            if r['condition']==v['condition']:assert not base.isomorphic(v['case'],r['case'])
    update_groups={};validation_rows=[];checkpoint_files=[]
    for seed in cfg['train_seeds']:
        initials=[]
        for arm in cfg['arms']:
            check();folder=RESULT/'jobs'/f'{arm}_{seed}'
            updates=[json.loads(l) for l in (folder/'updates.jsonl').read_text().splitlines()]
            assert [r['update'] for r in updates]==list(range(1,6001))
            assert all(r['forward_count']==2 and r['optimizer_updates']==1 and abs(r['loss']-.5*sum(p['loss'] for p in r['parts']))<1e-12 for r in updates)
            assert all(r['condition'] in conditions['training'][arm] for r in updates)
            update_groups[arm,seed]=updates
            state=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True);initials.append(state)
            for point in cfg['validation_updates']:
                v=read(folder/f'validation_u{point:06d}.json');validation_rows.append(dict(arm=arm,seed=seed,update=point,**v))
            for line in (folder/'checkpoint_index.jsonl').read_text().splitlines():
                check();r=json.loads(line);path=RESULT/r['path'];assert sha(path)==r['sha256']
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['completed_updates']==r['completed_updates'] and ck['completed_updates'] in (1000,3000,6000)
                assert ck['binding']==read(RESULT/'freeze.json') and ck['budget']==read(RESULT/'activation.json')
                assert all(int(s['step'])==ck['completed_updates'] for s in ck['optimizer']['state'].values())
                if r['completed_updates']==6000:
                    final=torch.load(folder/'final6000.pt',map_location='cpu',weights_only=True)
                    assert all(torch.equal(t,final[k]) for k,t in ck['model'].items())
                checkpoint_files.append(r);del ck
        assert initials[0].keys()==initials[1].keys()
        assert all(torch.equal(initials[0][k],initials[1][k]) for k in initials[0])
        left,right=update_groups['narrow',seed],update_groups['broad',seed]
        for a,b in zip(left,right):
            for k in ('condition_uniform','augmentation_seed','noise_seeds','t','query_sha256'):assert a[k]==b[k]
        del initials
    assert sum(map(len,update_groups.values()))==36000 and len(checkpoint_files)==18
    save(RESULT/'validation_curves.json',validation_rows)
    # All newly generated chains, never a historical substitute.
    rows=[read(p) for p in sorted(RESULT.glob('jobs/*/generated/*_result.json'))]
    assert len(rows)==528
    expected=set(itertools.product(cfg['arms'],cfg['train_seeds'],conditions['final_targets'],cfg['sampling_seeds']))
    assert {(r['arm'],r['seed'],r['condition'],r['sampling_seed']) for r in rows}==expected
    paired={};graphs={};groups=defaultdict(list)
    case=metadata_case(base,PANEL,targets)
    for i,r in enumerate(rows):
        check();assert sha(RESULT/r['graph_file'])==r['graph_sha256'] and sha(RESULT/r['steps_file'])==r['steps_sha256']
        assert r['target_fingerprint']==targets[r['condition']].fingerprint and r['newly_executed_chain']
        steps=read(RESULT/r['steps_file']);assert [s['t'] for s in steps]==list(range(64,0,-1))
        assert all(s['exact_coverage'] and s['blocks']==4 and s['covered_slots']==312 for s in steps)
        random=(r['initial_state_sha256'],[(s['uniform_sha256'],s['query_sha256'],s['rng_after']) for s in steps])
        if r['sampling_seed'] in paired:assert paired[r['sampling_seed']]==random
        else:paired[r['sampling_seed']]=random
        G=independent(r,targets);graphs[r['graph_file']]=G
        with np.load(RESULT/r['graph_file'],allow_pickle=False) as z:
            f,loc=case['space'].encode(z['edges'].T);state=np.zeros(case['space'].total,np.uint8)
            state[case['space'].offsets[f]+loc]=1
            assert base.sha(state.tolist())==steps[-1]['state_sha256']
        groups[r['arm'],r['seed'],r['condition']].append(r)
        if i%50==0:print('independent graph/path audit',i,'/528',flush=True)
    summaries=[];classes=[]
    for (arm,seed,name),group in sorted(groups.items()):
        check();group.sort(key=lambda r:r['sampling_seed']);assert len(group)==8
        reps=[];freq=[];assign=[]
        for r in group:
            G=graphs[r['graph_file']];h=nx.weisfeiler_lehman_graph_hash(G,node_attr='type',edge_attr='relation');found=None
            for j,(other,oh) in enumerate(reps):
                if h==oh and nx.is_isomorphic(G,other,node_match=lambda x,y:x['type']==y['type'],edge_match=lambda x,y:x['relation']==y['relation']):found=j;break
            if found is None:found=len(reps);reps.append((G,h));freq.append(0)
            freq[found]+=1;assign.append(dict(sampling_seed=r['sampling_seed'],local_class=found))
        ff=[features(r) for r in group];tv=sorted(v[8] for v,n in targets[name].roles[3] for _ in range(n))[::-1]
        ranks=[r['target_errors']['Genre']['degrees_descending'] for r in group]
        summaries.append(dict(arm=arm,seed=seed,condition=name,category=category(name,conditions),graphs=8,
            statistics={k:stat([f[k] for f in ff]) for k in ff[0]},
            G_sorted_degree_mean=np.mean(ranks,axis=0).tolist(),G_target_sorted=tv,
            G_signed_rank_bias=np.mean(np.asarray(ranks)-tv,axis=0).tolist(),
            G_mean_histogram=np.mean([np.bincount(x,minlength=19) for x in ranks],axis=0).tolist(),
            topology_classes=len(freq),topology_frequencies=freq,effective_classes=math.exp(-sum(n/8*math.log(n/8) for n in freq))))
        classes.append(dict(arm=arm,seed=seed,condition=name,assignments=assign,frequencies=freq))
    differences=[]
    for seed in cfg['train_seeds']:
        for name in conditions['final_targets']:
            a=next(r for r in summaries if (r['arm'],r['seed'],r['condition'])==('narrow',seed,name))
            b=next(r for r in summaries if (r['arm'],r['seed'],r['condition'])==('broad',seed,name))
            differences.append(dict(seed=seed,condition=name,category=category(name,conditions),
                mean_broad_minus_narrow={k:b['statistics'][k]['mean']-a['statistics'][k]['mean'] if a['statistics'][k]['mean'] is not None and b['statistics'][k]['mean'] is not None else None for k in a['statistics']}))
    save(RESULT/'generation.json',rows);save(RESULT/'per_seed.json',dict(summaries=summaries,paired_differences=differences))
    save(RESULT/'topology_classes.json',classes)
    counts=[dict(arm=arm,seed=seed,updates=dict(Counter(r['condition'] for r in u)),
                 samples={k:2*v for k,v in Counter(r['condition'] for r in u).items()}) for (arm,seed),u in update_groups.items()]
    save(RESULT/'training_condition_counts.json',counts)
    save(RESULT/'audit.json',dict(status='passed',updates=36000,checkpoint_count=18,graphs=528,
        reverse_steps=528*64,sampling_network_forwards=528*64*4,wall_seconds=time.monotonic()-start,
        checks=['source/data/complete target binding','initial482 tensors exactly paired','continuous optimizer steps/checkpoints/RNG',
                'condition/time/query random sources paired; noisy inputs NOT claimed identical','all train/validation topology exclusions',
                'all528 raw graphs with independent integer degree/joint/W1/connectivity/typed triangles/DMA closure',
                'independent node-simple MP1/2/3 full pattern counts','all64step coverage/RNG/final-state links','exact typed topology classes per8draw cell'],
        GPU_used=False,postprocessing=False,isolates_not_a_gate=True))

if __name__=='__main__':audit()
