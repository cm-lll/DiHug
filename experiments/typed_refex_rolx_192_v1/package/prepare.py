"""Static intake and single frozen unsupervised basis fit; no diffusion fitting."""
import hashlib,itertools,json,pathlib,time
import numpy as np
import networkx as nx
from hsgd.bw_preflight.common import make_case,plan_for,query_inventory
from features import SPEC,extract,fit_basis,encode

ROOT=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,obj):pathlib.Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False))

def load_case(name):
    cfg=json.loads((ROOT/'config.json').read_text());entry=cfg['datasets'][name]
    p=ROOT/entry['file']
    if sha(p)!=entry['sha256']:raise ValueError('dataset binding mismatch')
    with np.load(p,allow_pickle=False) as z:
        c=make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        if not np.array_equal(c['space'].encode(z['edges'].T)[0],z['relations'].astype(np.int64)-1):raise ValueError('relation convention')
    return c,cfg

def main():
    received=ROOT.parent/'received/subgraph_refined_20261027';out=ROOT/'assets';out.mkdir(exist_ok=False)
    manifest=json.loads((received.parent/'PACKAGE_MANIFEST.json').read_text());verified=0
    for row in manifest['files']:
        p=received/row['path']
        if not p.resolve().is_relative_to(received.resolve()):raise ValueError('manifest traversal')
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
        verified+=1
    original=json.loads((ROOT/'source_config.json').read_text())
    cfg={k:original[k] for k in ['model','optimizer','precision','T','rounds','schedule_offset','tf32']}
    cfg.update(version='typed_refex_rolx_192_v1',arms=['A','B','C'],formal_authorized=False,
               feature_spec=SPEC,inputs_per_update=1,engineering_only=True,max_updates_per_dataset_arm=5,
               seed=2026102709,noise_times=[1,500,1000],threads=1,
               tolerances={'float64_features':1e-10,'fp32_atol':2e-6,'fp32_rtol':1e-5},
               datasets={},query_contract='original family stratified 20 blocks, active reference-positive families; R/S sum CE',
               source_note='model/optimizer from E07 historical_config; single input engineering update, not old normal2 timing')
    result={'manifest_files_verified':verified,'datasets':{}}
    for name,expected in [('IMDB',(323,[48,56,78,10],3,188)),('PubMed',(926,[40,74,66,12],1,192))]:
        path=received/'samples'/name/'graph.npz';meta=json.loads((path.parent/'metadata.json').read_text())
        assert sha(path)==meta['graph_sha256']
        with np.load(path,allow_pickle=False) as z:arr={k:z[k] for k in z.files}
        types,edges,rel,ids=arr['types'],arr['edges'],arr['relations'],arr['original_ids']
        refpath=received/'inputs/reference'/(name+'.npz');assert sha(refpath)==meta['original_graph_sha256']
        with np.load(refpath,allow_pickle=False) as z:ref={k:z[k] for k in z.files}
        assert types.shape==(192,) and len(np.unique(ids))==192 and np.array_equal(ids,np.load(path.parent/'original_node_ids.npy'))
        assert np.array_equal(ref['types'][ids],types) and np.array_equal(ref['schema'],arr['schema'])
        assert edges.shape==(expected[0],2) and np.array_equal(np.bincount(types,minlength=4),expected[1])
        assert np.array_equal(edges,edges[np.lexsort((edges[:,1],edges[:,0]))])
        reverse=np.full(len(ref['types']),-1,dtype=np.int64);reverse[ids]=np.arange(192)
        selected=(reverse[ref['edges']]>=0).all(1);induced=reverse[ref['edges'][selected]]
        assert np.array_equal(induced,edges) and np.array_equal(ref['relations'][selected],rel),'not exact induced graph'
        c=make_case(types,edges.T,tuple(map(tuple,arr['schema'])))
        assert np.array_equal(c['space'].encode(edges.T)[0]+1,rel)
        g=nx.Graph();g.add_nodes_from(range(192));g.add_edges_from(edges)
        assert nx.number_of_selfloops(g)==0 and g.number_of_edges()==len(edges) and len(list(nx.isolates(g)))==0
        comps=sorted(map(len,nx.connected_components(g)),reverse=True);assert (len(comps),comps[0])==expected[2:]
        # Store only network-safe data. Original IDs stay in read-only intake.
        final=out/(name+'.npz');np.savez_compressed(final,types=types,edges=edges,relations=rel,schema=arr['schema'])
        cfg['datasets'][name]={'file':str(final.relative_to(ROOT)),'sha256':sha(final),'received_sha256':sha(path),
                               'reference_sha256':sha(refpath),'type_names':json.loads((received/'source_preview/inputs/recompute/source_release/configs'/(name+'.json')).read_text())['data']['type_names']}
        start=time.perf_counter();f=extract(types,edges.T,rel.astype(np.int64)-1,arr['schema']);feature_sec=time.perf_counter()-start
        basis,receipt=fit_basis(f);coef,kkt=encode(f,basis)
        bp=out/(name+'_basis.npz');np.savez_compressed(bp,basis=basis)
        cfg['datasets'][name].update(basis_file=str(bp.relative_to(ROOT)),basis_sha256=sha(bp))
        result['datasets'][name]=dict(nodes=192,edges=len(edges),type_counts=expected[1],components=comps,
               source_sha256=sha(path),exact_induced=True,original_ids_network_input=False,
               query=query_inventory(c,plan_for(c,2026102709,20)),feature_shape=list(f.shape),feature_seconds=feature_sec,
               basis=receipt,encoding=kkt,wedges=int(sum(d*(d-1)//2 for _,d in g.degree())))
    save(ROOT/'config.json',cfg);save(ROOT/'data_audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
