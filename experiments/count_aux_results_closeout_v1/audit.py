"""Independent checks of report identities/arithmetic and copied graph mappings."""
import argparse, collections, json, math, pathlib, time
import networkx as nx
import numpy as np
from build import read, save, sha, ident, jobpath, tag, SEEDS, SAMPLES

def near(a,b):
    assert (a is None)==(b is None),(a,b)
    if a is not None:assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-12),(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);out=pathlib.Path(ap.parse_args().output)
    begin=time.time();doc=read(out/'evaluation.json');idx={ident(r):r for r in doc['records']};checks=collections.Counter();proof=[]
    assert len(idx)==len(doc['records'])==96
    for key,r in idx.items():
        base=out/'run'/jobpath(r);z=np.load(base/'graph.npz',allow_pickle=False);types=z['types'];edges=z['edges'];schema=z['schema']
        orig=nx.Graph();orig.add_nodes_from(range(len(types)));orig.add_edges_from(edges.T)
        if r['version']=='lcc':
            path=out/'run/derived_lcc_evaluation'/tag(r);lz=np.load(path/'lcc.npz',allow_pickle=False);mapping=lz['new_to_original']
            expected=sorted([sorted(c) for c in nx.connected_components(orig)],key=lambda c:(-len(c),c))[0]
            assert mapping.tolist()==expected
            assert set(map(tuple,mapping[lz['edges']].T))=={tuple(e) for e in edges.T if int(e[0]) in expected and int(e[1]) in expected}
            np.testing.assert_array_equal(lz['types'],types[mapping]);np.testing.assert_array_equal(lz['schema'],schema)
            assert sha(path/'lcc.npz')==read(path/'mapping.json')['derived_sha256']
            types=lz['types'];edges=lz['edges'];checks['lcc_exact_induced_mapping']+=1
        else:assert sha(base/'graph.npz')==r['graph_source_sha256']
        g=nx.Graph();g.add_nodes_from(range(len(types)));g.add_edges_from(edges.T)
        raw=r['raw'];assert nx.number_of_selfloops(g)==0 and g.number_of_edges()==edges.shape[1]
        assert raw['nodes']==len(types) and raw['edges']==g.number_of_edges()
        assert raw['triangles']==sum(nx.triangles(g).values())//3
        assert raw['components']==nx.number_connected_components(g)
        assert raw['lcc']==max(map(len,nx.connected_components(g)))
        lookup={tuple(x):i for i,x in enumerate(schema)};counts=collections.Counter(lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T)
        for i in range(10):assert raw[f'family{i}_edges']==counts[i]
        for k,v in r['absolute_error'].items():
            a,b=raw[k],r['reference'].get(k);near(v,abs(a-b) if a is not None and b is not None else None);checks['absolute_error_per_graph']+=1
        d=r['distribution_error'];near(d['all10_compat_W1'],.4*d['main_incident4_W1'] if d['main_incident4_W1'] is not None else None)
        assert sum(v['path_denominator'] for v in r['statistics']['typed_twohop_closure'].values())==raw['wedge_count']
        assert sum(v['path_numerator'] for v in r['statistics']['typed_twohop_closure'].values())==3*raw['triangles']
        for u in (1,2,3):assert raw[f'MP{u}_total']==r['statistics']['mp'][u-1]['total']
        checks['graph_identity_counts_labels_w1_closure']+=1;proof.append(dict(identity=key,passed=True))
    for row in doc['cell_means']:
        vals=[idx.get((row['dataset'],row['train_seed'],row['arm'],row['checkpoint'],s,row['version']),{}).get(row['group'],{}).get(row['metric']) for s in SAMPLES]
        near(row['value'],sum(vals)/2 if all(v is not None for v in vals) else None);checks['strict_two_sample_means']+=1
    for row in doc['paired']:
        def get(arm):return idx.get((row['dataset'],row['train_seed'],arm,row['checkpoint'],row['sample_seed'],row['version']),{}).get(row['group'],{}).get(row['metric'])
        a,b=get('CE_COUNT'),get('CE');near(row['difference'],a-b if a is not None and b is not None else None);checks['same_seed_same_stage_pairs']+=1
    for row in doc['length_paired']+doc['total_length_paired']:
        def get(u):return idx.get((row['dataset'],row['train_seed'],row['arm'],u,row['sample_seed'],row['version']),{}).get(row['group'],{}).get(row['metric'])
        a,b=get(row['later']),get(row['earlier']);near(row['difference'],a-b if a is not None and b is not None else None);checks['same_trajectory_length_pairs']+=1
    inputs=read(out/'input_pairing.json');assert all(r['paired_inputs_exact'] and r['complete_updates'] for r in inputs)
    assert all(x['consistent'] for x in read(out/'validation_input_consistency.json'))
    # Reference totals/schema are unchanged; graph count is not doubled by LCC.
    inv=read(out/'material_inventory.json');assert len(inv)==48
    expected={('IMDB',3000):8,('IMDB',6000):8,('PubMed',1000):8,('PubMed',2000):8,('PubMed',3000):8,('PubMed',6000):8}
    assert collections.Counter((r['dataset'],r['checkpoint']) for r in inv)==expected
    assert not doc['missing_identities']
    save(out/'independent_audit.json',dict(status='passed',records=96,unique_original_graphs=48,counts=[dict(dataset=d,checkpoint=u,count=n) for (d,u),n in expected.items()],checks=checks,per_graph=proof,seconds=time.time()-begin,new_model_forwards=0,new_training_updates=0,new_chains=0))
    print(json.dumps(dict(status='passed',checks=checks,seconds=time.time()-begin)))
if __name__=='__main__':main()
