"""CPU-only read-only independent checks of completed graphs and recorded contracts."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import networkx as nx

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=1e-11),(a,b)

def stats(types,edges,schema):
    edges=np.asarray(edges)
    assert edges.ndim==2 and edges.shape[0]==2
    assert np.all(edges[0]<edges[1]) and edges.min()>=0 and edges.max()<len(types)
    assert len(set(map(tuple,edges.T)))==edges.shape[1]
    lookup={tuple(map(int,p)):i for i,p in enumerate(schema)}
    fam=np.array([lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges.T])
    deg=np.zeros((len(types),len(schema)),dtype=np.int64)
    np.add.at(deg,(edges[0],fam),1);np.add.at(deg,(edges[1],fam),1)
    g=nx.Graph();g.add_nodes_from(range(len(types)));g.add_edges_from(edges.T)
    tr=Counter()
    for u in g:
        for v in g[u]:
            if v<=u:continue
            for w in set(g[u])&set(g[v]):
                if w>v:tr[str(tuple(sorted(map(int,types[[u,v,w]]))))]+=1
    comps=sorted(map(len,nx.connected_components(g)),reverse=True)
    # Number of unoriented simple length-3 paths: central-edge products minus triangles.
    mp2=sum(d*(d-1)//2 for _,d in g.degree())
    mp3=sum((g.degree(u)-1)*(g.degree(v)-1)-len(set(g[u])&set(g[v])) for u,v in g.edges())
    return dict(deg=deg,fam=fam,edges=edges.shape[1],triangles=sum(tr.values()),typed_triangles=tr,
                lcc=comps[0],isolates=len(list(nx.isolates(g))),component_sizes=comps,
                average_clustering=nx.average_clustering(g),maximum_degree=max(dict(g.degree()).values()),
                mp=[edges.shape[1],mp2,mp3])

def errors(types,schema,deg,ref):
    w1=joint=0.
    for t in np.unique(types):
        mask=types==t;rs=[r for r,p in enumerate(schema) if t in p];vals=[]
        for r in rs:
            a,b=deg[mask,r],ref[mask,r];size=int(max(a.max(),b.max()))+1
            vals.append(float(np.abs(np.cumsum(np.bincount(a,minlength=size)/len(a)-np.bincount(b,minlength=size)/len(b))).sum()))
        a,b=Counter(map(tuple,deg[mask])),Counter(map(tuple,ref[mask]));n=int(mask.sum())
        weight=n/len(types);w1+=weight*np.mean(vals)
        joint+=weight*.5*sum(abs(a[k]/n-b[k]/n) for k in a.keys()|b.keys())
    return w1,joint

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.time();root=Path(a.run);src=Path(__file__).parent
    x=json.loads((root/'unified_evaluation.json').read_text());cfg=json.loads((src/'package/config.json').read_text())
    assert x['status']=='complete' and x['evaluated_graphs']==16
    graphs=[];cells={};payloads=[]
    for ds in ['IMDB','PubMed']:
        refpath=src/'package'/cfg['datasets'][ds]['file'];assert sha(refpath)==cfg['datasets'][ds]['sha256']
        with np.load(refpath,allow_pickle=False) as z:
            types,schema=z['types'],z['schema'];ref=stats(types,z['edges'].T,schema)
        init=[];inputs=[];rng=[]
        for arm in 'OABC':
            name=ds+'_'+arm;folder=root/name;receipt=json.loads((folder/'summary.json').read_text())
            assert receipt['status']=='complete' and receipt['completed_updates']==1000 and receipt['completed_chains']==2
            ini=json.loads((folder/'initialization.json').read_text());assert all(ini['copied_exact'].values());init.append(ini)
            rng.append(json.loads((folder/'training_rng_reset.json').read_text()))
            rows=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
            assert [r['update'] for r in rows]==list(range(1,1001))
            assert all(r['optimizer_steps']==1 and len(r['inputs'])==2 and math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in rows)
            inputs.append([r['input_sha256'] for r in rows])
            warm=[r['normal2_seconds'] for r in rows if not r['warmup']]
            cells[name]=dict(updates=1000,normal2_total_seconds=sum(r['normal2_seconds'] for r in rows),
                             measured_postwarmup_mean_seconds=float(np.mean(warm)),parameters=ini['parameters'],
                             peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib'],
                             sampling_seconds=[])
            for seed in [2026102821,2026102822]:
                s=folder/f'sample_{seed}';binding=json.loads((s/'binding.json').read_text());assert binding['graph_sha256']==sha(s/'graph.npz') and binding['checkpoint_sha256']==sha(folder/'final.pt')
                steps=[json.loads(l) for l in (s/'steps.jsonl').read_text().splitlines()]
                assert [r['time'] for r in steps]==list(range(1000,0,-1))
                assert all(r['exact_coverage'] and r['blocks']==1 and r['covered_slots']==(13120 if ds=='IMDB' else 18336) for r in steps)
                with np.load(s/'graph.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['types'],types);np.testing.assert_array_equal(z['schema'],schema)
                    m=stats(types,z['edges'],schema);np.testing.assert_array_equal(m['fam'],z['families'])
                old=x['cells'][name]['graphs'][str(seed)]['raw_metrics']
                for k in ['edges','triangles','lcc','isolates','maximum_degree','average_clustering']:close(m[k],old[k])
                assert m['component_sizes']==old['component_sizes'] and dict(m['typed_triangles'])==old['typed_triangles']
                np.testing.assert_array_equal(np.bincount(m['fam'],minlength=len(schema)),old['family_edges'])
                assert m['mp']==[v['total'] for v in old['mp']]
                w,j=errors(types,schema,m['deg'],ref['deg']);close(w,old['W1_definitions']['incident4']);close(.4*w,old['W1_definitions']['all10_compat']);close(j,old['primary']['type_weighted_joint_degree_TV'])
                graphs.append(dict(cell=name,seed=seed,status='passed',W1_incident4=w,joint_degree_TV=j,steps=1000,graph_sha256=sha(s/'graph.npz')))
                cells[name]['sampling_seconds'].append(binding['seconds'])
        assert all(v==inputs[0] for v in inputs) and all(v==rng[0] for v in rng)
        common=set.intersection(*(set(i['tensor_sha256']) for i in init))
        # A retains additional CN columns in the first joint-head matrix.
        # Its full-array hash must differ; compare the actual common column slice.
        assert all(all(i['tensor_sha256'][k]==init[0]['tensor_sha256'][k] for k in common if k!='joint_head.0.weight') for i in init)
        import torch
        weights=[]
        for arm in 'OABC':
            ck=torch.load(root/(ds+'_'+arm)/'initial.pt',map_location='cpu')
            assert ck['updates']==0
            weights.append(ck['model']['joint_head.0.weight'].clone())
            del ck
        width=min(v.shape[1] for v in weights)
        assert all(torch.equal(v[:,:width],weights[0][:,:width]) for v in weights)
    # Check every emitted checkpoint/state hash without deserializing model objects.
    for f in sorted(root.rglob('*.sha256.json')):
        ck=f.with_name(f.name.removesuffix('.sha256.json')+'.pt');expected=json.loads(f.read_text())['sha256']
        assert ck.exists() and sha(ck)==expected,str(ck)
    for f in sorted(root.rglob('*')):
        if f.is_file():payloads.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)))
    result=dict(status='passed',scope='independent CPU graph checks, receipt/hash checks; no network call, optimizer update or new graph',
                graphs=graphs,cells=cells,payloads=payloads,seconds=time.time()-start,
                limitations=['No additional GPU numerical test','Does not establish multi-training-seed stability or adequate population diversity'])
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],graphs=len(graphs),cells=len(cells),payloads=len(payloads),seconds=result['seconds'])))

if __name__=='__main__':main()
