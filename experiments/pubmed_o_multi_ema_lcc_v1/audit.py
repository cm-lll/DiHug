"""Independent mapping, induced-subgraph, capacity and summary arithmetic audit."""
import argparse,collections,itertools,json,math,pathlib,time,statistics
import numpy as np
import networkx as nx
from metrics import sha
def read(p):return json.loads(pathlib.Path(p).read_text())
def eq(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def paths(types,edges,fs,allow_repeats=False):
    adj=[{} for _ in types]
    for (u,v),f in zip(edges.T,fs):adj[u][int(v)]=int(f);adj[v][int(u)]=int(f)
    counters=[collections.Counter() for _ in range(3)]
    def dfs(nodes,sig):
        if len(nodes)>1:
            key=tuple(sig);key=min(key,key[::-1])
            if allow_repeats or nodes[0]<nodes[-1]:counters[len(nodes)-2][','.join(map(str,key))]+=1
        if len(nodes)==4:return
        for v,r in adj[nodes[-1]].items():
            if allow_repeats or v not in nodes:dfs(nodes+[v],sig+[r,int(types[v])])
    for u in range(len(types)):dfs([u],[int(types[u])])
    return counters
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--offline-review',action='store_true');args=parser.parse_args();root=pathlib.Path(args.root);start=time.time();data=read(root/'analysis.json');ledger=read(root/'ledger.json')
    def check():
        if args.offline_review:
            if time.time()-start>120:raise TimeoutError('offline read-only verification120seconds')
        elif time.time()>ledger['deadline_epoch']-90:raise TimeoutError('new45minute budget cutoff')
    manifest=root/'SHA256SUMS'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            check();expected,name=line.split('  ',1);target=(root/name).resolve();assert root.resolve() in target.parents;assert sha(target)==expected,name
    checks=0;by={(r['train_seed'],r['readout'],r['sample_seed'],r['version']):r for r in data['records']};sources=read(root/'input_manifest.json')
    for identity in sources:
        check();folder=root/'graphs'/('s%d'%identity['train_seed'])/identity['readout']/str(identity['sample_seed']);mapping=read(folder/'mapping.json')
        assert sha(folder/'original.npz')==identity['sha256'] and sha(folder/'lcc.npz')==identity['derived_sha256']
        with np.load(folder/'original.npz',allow_pickle=False) as z:types=z['types'];edges=z['edges'];families=z['families'];schema=z['schema']
        with np.load(folder/'lcc.npz',allow_pickle=False) as z:tt=z['types'];ee=z['edges'];ff=z['families'];ids=z['new_to_original'];old_to_new=z['old_to_new'];np.testing.assert_array_equal(z['schema'],schema)
        graph=nx.Graph();graph.add_nodes_from(range(len(types)));graph.add_edges_from(edges.T);cc=sorted([sorted(c) for c in nx.connected_components(graph)],key=lambda x:(-len(x),x));np.testing.assert_array_equal(ids,cc[0]);np.testing.assert_array_equal(tt,types[ids]);np.testing.assert_array_equal(old_to_new[ids],np.arange(len(ids)));assert set(old_to_new[np.setdiff1d(np.arange(len(types)),ids)])<=set([-1])
        expected={(u,v):int(r) for (u,v),r in zip(edges.T,families) if u in set(ids) and v in set(ids)};actual={(int(ids[u]),int(ids[v])):int(r) for (u,v),r in zip(ee.T,ff)};assert actual==expected
        for version,t,e,f in [('original',types,edges,families),('lcc',tt,ee,ff)]:
            record=by[identity['train_seed'],identity['readout'],identity['sample_seed'],version];s=record['statistics'];n=len(t);g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(e.T)
            assert n==s['nodes'] and e.shape[1]==s['edges'];counts=np.bincount(t,minlength=4);caps=[int(counts[a]*(counts[a]-1)//2 if a==b else counts[a]*counts[b]) for a,b in schema];assert caps==s['capacities'];assert [len(c) for c in sorted(nx.connected_components(g),key=len,reverse=True)]==s['component_sizes']
            np.testing.assert_array_equal(np.sort(t[e],axis=0).T,schema[f]);degree=np.zeros((n,10),int);np.add.at(degree,(e[0],f),1);np.add.at(degree,(e[1],f),1)
            for typ,name in enumerate(['Gene','Disease','Chemical','Species']):
                vals=degree[t==typ].sum(1);sp=s['typed_degree'][name];assert sorted(map(int,vals),reverse=True)==sp['sorted_degrees'];assert sp['count']==len(vals)
            eq(nx.average_clustering(g),s['average_clustering']);eq(sum(nx.triangles(g).values())/3,s['triangles'])
            common_cycles=sum(math.comb(len(set(g[u])&set(g[v])),2) for u in range(n) for v in range(u+1,n))//2;assert common_cycles==s['four_cycles']
            checks+=10
        # Saved raw frozen paths = LCC paths + exact tiny excluded-component paths.
        original=by[identity['train_seed'],identity['readout'],identity['sample_seed'],'original']['statistics'];kept=by[identity['train_seed'],identity['readout'],identity['sample_seed'],'lcc']['statistics'];removed=[collections.Counter() for _ in range(3)];removed_walk=[collections.Counter() for _ in range(3)]
        for c in cc[1:]:
            sub=graph.subgraph(c);local={old:i for i,old in enumerate(c)};ed=np.array([[local[u],local[v]] for u,v in sub.edges()],int).reshape(-1,2).T;ft=np.array([families[next(i for i,(a,b) in enumerate(edges.T) if (a,b)==(min(u,v),max(u,v)))] for u,v in sub.edges()],int)
            for target,allow in [(removed,False),(removed_walk,True)]:
                p=paths(types[c],ed,ft,allow)
                for i in range(3):target[i].update(p[i])
        for field,remove in [('mp',removed),('typed_walks',removed_walk)]:
            for i in range(3):
                total=collections.Counter(kept[field][i]['counts']);total.update(remove[i]);assert +total==collections.Counter(original[field][i]['counts']);checks+=1
    # Every EMA pair: differences are formed on per-graph absolute errors first.
    import csv
    pairs=list(csv.DictReader((root/'ema_pairs.csv').open()));cohorts=list(csv.DictReader((root/'cohort_pairs.csv').open()))
    for row in pairs:
        check();seed=int(row['train_seed']);kind=row['readout'];version=row['version'];group=row['metric_group'];metric=row['metric'];values=[]
        for sample in [2026102821,2026102822]:
            a,b=by[seed,kind,sample,version][group].get(metric),by[seed,'raw',sample,version][group].get(metric);values.append(None if a is None or b is None else a-b)
        expected=None if any(x is None for x in values) else sum(values)/2;eq(None if row['seed_difference']=='' else float(row['seed_difference']),expected);checks+=1
    index=collections.defaultdict(list)
    for row in pairs:index[row['readout'],row['version'],row['metric_group'],row['metric']].append(row)
    for row in cohorts:
        selected=[r for r in index[row['readout'],row['version'],row['metric_group'],row['metric']] if row['cohort']=='all_eight' or (r['cohort']==row['cohort'])];values=[None if r['seed_difference']=='' else float(r['seed_difference']) for r in selected];expected=None if any(v is None for v in values) else sum(values)/len(values);eq(None if row['mean_difference']=='' else float(row['mean_difference']),expected);checks+=1
        eq(None if row['median_difference']=='' else float(row['median_difference']),None if expected is None else statistics.median(values));checks+=1
        if expected is not None:assert [int(row[k]) for k in ['negative_count','zero_count','positive_count']]==[sum(v<0 for v in values),sum(v==0 for v in values),sum(v>0 for v in values)]
        else:assert all(row[k]=='' for k in ['negative_count','zero_count','positive_count'])
        assert int(row['valid_seeds'])==sum(v is not None for v in values);checks+=1
    for row in csv.DictReader((root/'lcc_minus_original.csv').open()):
        seed=int(row['train_seed']);kind=row['readout'];sample=int(row['sample_seed']);group=row['metric_group'];key=row['metric']
        a,b=by[seed,kind,sample,'lcc'][group].get(key),by[seed,kind,sample,'original'][group].get(key)
        eq(None if row['difference']=='' else float(row['difference']),None if a is None or b is None else a-b);checks+=1
    for record in data['records']:
        for key,value in record['absolute_error'].items():
            a,b=record['raw'][key],record['reference'][key];eq(value,None if a is None or b is None else abs(a-b));checks+=1
    result=dict(status='passed',seconds=time.time()-start,graphs=64,graph_versions=128,mapping_type_relation_no_added_edges_checks=True,actual_population_and_capacity_checks=True,excluded_component_path_additivity_all_lengths=True,reference_fixed_weights_and_all10_checked_by_analysis=True,arithmetic_and_structural_checks=checks,model_imported=False,training=0,sampling=0)
    output=root/('offline_audit_'+time.strftime('%Y%m%dT%H%M%S')+'.json' if args.offline_review else 'audit.json')
    if args.offline_review and output.exists():raise FileExistsError(output)
    result['offline_review']=args.offline_review;output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
