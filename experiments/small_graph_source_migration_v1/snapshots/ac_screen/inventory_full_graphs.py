"""CPU-only static source/data inventory; never imports graph models or fits bases."""
import argparse,pathlib,json,hashlib,collections,math
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=pathlib.Path(a.source);out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    inventory=json.loads((src/'role_target_scalability_v1/evidence/large_graph_inventory.json').read_text());rows={}
    for dataset in ('IMDB','PubMed'):
        path=src/'joint_cn_dual_rounds_v1_1/package/data'/(dataset+'.npz');cp=src/'joint_cn_dual_rounds_v1_1/package/configs'/(dataset+'.json');cfg=json.loads(cp.read_text());expected=cfg['data']['sha256']
        assert sha(path)==expected==inventory[dataset]['sha256']
        with np.load(path,allow_pickle=False) as z:types=z['types'];edges=z['edges'];families=z['relations'].astype(int)-1;schema=z['schema']
        n=len(types);sizes=np.bincount(types,minlength=4);counts=np.bincount(families,minlength=10)
        assert edges.shape[1]==2 and (edges[:,0]<edges[:,1]).all() and np.unique(edges,axis=0).shape==edges.shape
        assert edges.min()>=0 and edges.max()<n and np.array_equal(np.sort(types[edges],axis=1),schema[families])
        assert np.array_equal(edges,edges[np.lexsort((edges[:,1],edges[:,0]))])
        assert (n,len(edges))==((1653,4079) if dataset=='IMDB' else (1565,12284))
        caps=np.array([sizes[x]*(sizes[x]-1)//2 if x==y else sizes[x]*sizes[y] for x,y in schema]);prior=counts/caps;active=(prior>0)&(prior<1)
        deg=np.zeros((n,10),dtype=int);np.add.at(deg,(edges[:,0],families),1);np.add.at(deg,(edges[:,1],families),1)
        roles=[len(set(map(tuple,deg[types==t]))) for t in range(4)]
        assert sizes.tolist()==cfg['data']['type_counts'] and counts.tolist()==cfg['data']['family_counts']
        # Original per-family split yields ceil/floor counts per round.
        blocks=[int(sum((cap//20+int(i<cap%20)) for cap,on in zip(caps,active) if on)) for i in range(20)]
        rows[dataset]=dict(path=str(path.resolve()),sha256=expected,bytes=path.stat().st_size,config_path=str(cp.resolve()),config_sha256=sha(cp),
            nodes=n,edges=len(edges),type_names=cfg['data']['type_names'],type_counts=sizes.tolist(),schema=schema.tolist(),family_edges=counts.tolist(),family_capacities=caps.tolist(),prior=prior.tolist(),active_slots=int(caps[active].sum()),schema_slots=int(caps.sum()),exact_joint_degree_kinds=roles,
            kernel=dict(T=cfg['T'],rounds=cfg['rounds'],schedule_offset=cfg['schedule_offset']),query_block_sizes=blocks,model=cfg['model'],
            static_memory_bytes=dict(refex_fp64=n*630*8,refex_fp32_gpu=n*630*4,seven_state_feature_panel=n*630*8*7,
                input_degree_fp64=n*10*8,adjacency_CSR_lower_bound=(2*len(edges)*12+(n+1)*4),max_candidate_CPU_arrays=caps.sum().item()*32,
                NNLS_basis_K8_upper_example=32*630*8,NNLS_coeff_K8_upper_example=n*32*8),
            operation_estimate=dict(refex_egonet_edge_checks=n*len(edges),refex_relation_aggregation_multiply_adds=2*len(edges)*30,nnls_solves_per_state=n,nnls_role_K='not chosen for full graph',nnls_note='per-type K with630 columns; convergence/data dependent, cannot extrapolate time by nodes only'),
            no_new_feature_extraction=True,no_model_loaded=True,full_graph_dictionary_available=False,
            dataset_identity='typed Gene/Disease/Chemical/Species graph; NOT PubMed23 paper retrieval corpus' if dataset=='PubMed' else 'typed Director/Actor/Movie/Genre graph')
    out.write_text(json.dumps(dict(datasets=rows,static_only=True,full_graph_training=0,full_graph_forward=0,dictionary_fits=0),indent=2))
if __name__=='__main__':main()
