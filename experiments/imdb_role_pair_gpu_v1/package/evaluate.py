"""Saved raw outputs only, train seed is the statistical replication unit."""
import itertools
from pathlib import Path
import numpy as np
import networkx as nx
import pair_core as p
def tails(metrics,reference):
    out={}
    for typ,role_counts in metrics['joint_relation_degree'].items():
        expand=lambda values:np.asarray([sum(map(int,k.split(','))) for k,n in values.items() for _ in range(n)])
        ref=expand(reference['joint_relation_degree'][typ]);actual=expand(role_counts);threshold=max(1,int(np.ceil(np.quantile(ref,.99))))
        relation=[]
        for a,b in zip(metrics['typed_degree'][typ]['by_relation'],reference['typed_degree'][typ]['by_relation']):
            threshold_r=max(1,int(np.ceil(b['p99'])));hist=np.asarray(a['histogram'])
            relation.append({'threshold_ge_reference_P99':threshold_r,'high_count':int(hist[threshold_r:].sum())})
        out[typ]={'reference_P99_total_degree_threshold_ge':threshold,'high_count':int((actual>=threshold).sum()),'zero':int((actual==0).sum()),
            'low_le1':int((actual<=1).sum()),'sorted_top10_total_degree':sorted(map(int,actual),reverse=True)[:10],'by_relation':relation}
    return out
def aggregate(out,cfg,check=lambda:None):
    rows=[];groups={};case,_=p.dataset(cfg);reference=p.read(Path(out)/'reference_metrics.json')
    for seed in cfg['train_seeds']:
        for update in cfg['evaluation_updates']:
            for arm in cfg['arms']:
                graphs=[];key=f'{seed}/{update}/{arm}';values=[]
                for sample_seed in cfg['sampling_seeds']:
                    check();folder=Path(out)/f'seed{seed}'/f'samples_u{update:08d}'/arm/f'seed{sample_seed}'
                    summary=p.read(folder/'summary.json')
                    if summary['status']!='completed' or summary['graph_sha256']!=p.sha(folder/'graph.npz'):raise ValueError('missing/invalid final sample')
                    m=p.read(folder/'metrics.json');c=p.read(folder/'comparison.json')
                    with np.load(folder/'graph.npz',allow_pickle=False) as z:
                        types=z['types'];edges=z['edges'];f=z['families']
                    ef,local=case['space'].encode(edges)
                    if not np.array_equal(types,case['space'].types) or not np.array_equal(ef,f) or (edges[0]>=edges[1]).any() or len(np.unique(case['space'].offsets[ef]+local))!=edges.shape[1]:raise ValueError('independent raw graph/schema audit')
                    d=np.zeros((len(types),case['n_families']),int);np.add.at(d,(edges[0],f),1);np.add.at(d,(edges[1],f),1)
                    if m['family_edges']!=np.bincount(f,minlength=case['n_families']).tolist():raise ValueError('saved family metric differs')
                    from collections import Counter
                    for typ,name in enumerate(cfg['data']['type_names']):
                        joint={','.join(map(str,v)):n for v,n in Counter(map(tuple,d[types==typ])).items()}
                        if joint!=m['joint_relation_degree'][name]:raise ValueError('saved joint-degree metric differs')
                    G=nx.Graph();G.add_nodes_from((i,{'type':int(t)}) for i,t in enumerate(types));G.add_edges_from((int(a),int(b),{'relation':int(r)}) for (a,b),r in zip(edges.T,f))
                    wl=nx.weisfeiler_lehman_graph_hash(G,node_attr='type',edge_attr='relation');edge_set=set(map(tuple,edges.T.tolist()));graphs.append((wl,edge_set))
                    signed=c['family_edge_difference'];c['family_edge_absolute_error']=list(map(abs,signed))
                    row={'train_seed':seed,'checkpoint':update,'arm':arm,'sample_seed':sample_seed,'metrics':m,'comparison':c,'tails':tails(m,reference),'WL_hash':wl,'labelled_edge_sha256':p.digest(edges)};rows.append(row);values.append(row)
                groups[key]={'n':len(values),'train_seed':seed,'checkpoint':update,'arm':arm,
                    'mean_family_edge_signed_error':np.mean([v['comparison']['family_edge_difference'] for v in values],axis=0).tolist(),
                    'mean_family_edge_absolute_error':np.mean([v['comparison']['family_edge_absolute_error'] for v in values],axis=0).tolist(),
                    'mean_relation_degree_W1':{t:np.mean([v['comparison']['typed_relation_degree_W1'][t] for v in values],axis=0).tolist() for t in cfg['data']['type_names']},
                    'mean_joint_TV':{t:float(np.mean([v['comparison']['joint_relation_degree_TV'][t] for v in values])) for t in cfg['data']['type_names']},
                    'WL_distinct_lower_bound':len({g[0] for g in graphs}),
                    'labelled_unique':len({tuple(sorted(g[1])) for g in graphs}),
                    'mean_labelled_edge_symmetric_difference':float(np.mean([len(a[1]^b[1]) for a,b in itertools.combinations(graphs,2)])),
                    'equal_WL_not_claimed_isomorphic':True,'eight_graphs_not_independent_training_repeats':True}
    p.save(Path(out)/'all_graph_metrics.json',rows);p.save(Path(out)/'per_train_seed.json',groups)
    pairs={}
    for update in cfg['evaluation_updates']:
        pairs[str(update)]={}
        for t in cfg['data']['type_names']:
            diffs=[np.asarray(groups[f'{s}/{update}/role']['mean_relation_degree_W1'][t])-groups[f'{s}/{update}/base']['mean_relation_degree_W1'][t] for s in cfg['train_seeds']]
            pairs[str(update)][t]={'role_minus_base_per_seed':[x.tolist() for x in diffs],'equal_train_seed_mean':np.mean(diffs,axis=0).tolist()}
    p.save(Path(out)/'paired_summary.json',{'degree_W1':pairs,'unit':'three paired train seeds','same_reference_conditioned_only':True,'all_outputs_retained':True})
