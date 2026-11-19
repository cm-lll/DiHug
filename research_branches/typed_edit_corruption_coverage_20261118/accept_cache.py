"""Replay corruption with edge sets, count triangles independently, and rebuild features."""
from cache import *


def main():
    b.setup();verify();assert b.read(OUT/'cache_completion.json')['completed']
    ref=b.graph('pubmed');names=sorted(ref.family_specs);rc=metrics.counts(ref)
    rows=[];feature_error=0.;checked=0;tic=time.monotonic()
    for tid in range(12):
        for kind in ['U','B']:
            if kind=='B' and tid%2:continue
            folder=OUT/f'cache/{kind}_{tid}';saved=np.load(folder/'data.npz');r=b.read(folder/'record.json')
            edges={f:{tuple(sorted(e)) for e in ref.family_edges[f]} for f in names}
            nb=[set() for _ in range(ref.num_nodes)]
            for es in edges.values():
                for u,v in es:nb[u].add(v);nb[v].add(u)
            record_index=0;triangle_delta=0
            for entry in r['path']:
                edit=entry['edit'];f=names[edit['family'][0]];previous=edges[f].copy() if entry['step']%4==0 else None
                delta=0;degree={}
                for field,valid,sign in [('old_edges','old_valid',-1),('new_edges','new_valid',1)]:
                    for pair,yes in zip(edit[field][0],edit[valid][0]):
                        if not yes:continue
                        u,v=sorted(pair);assert u!=v
                        delta+=sign*len(nb[u]&nb[v])
                        if sign<0:
                            assert (u,v) in edges[f];edges[f].remove((u,v));nb[u].remove(v);nb[v].remove(u)
                        else:
                            assert (u,v) not in edges[f];edges[f].add((u,v));nb[u].add(v);nb[v].add(u)
                        degree[u]=degree.get(u,0)+sign;degree[v]=degree.get(v,0)+sign
                assert all(v==0 for v in degree.values())
                assert delta==entry['forward_delta'];triangle_delta+=delta;checked+=1
                if entry['step']%4:continue
                row=r['records'][record_index];assert row['step']==entry['step']
                y=row['label'];cand=row['candidates']
                assert cand['family'][y]==edit['family'][0] and cand['action'][y]==4
                assert row['inverse_delta']==-delta
                trial=edges[f].copy()
                for pair,yes in zip(cand['old_edges'][y],cand['old_valid'][y]):
                    if yes:assert tuple(sorted(pair)) in trial;trial.remove(tuple(sorted(pair)))
                for pair,yes in zip(cand['new_edges'][y],cand['new_valid'][y]):
                    if yes:assert tuple(sorted(pair)) not in trial;trial.add(tuple(sorted(pair)))
                assert trial==previous
                graph=b.TypedGraph(ref.node_types.copy(),ref.type_names,ref.family_specs,{f:sorted(es) for f,es in edges.items()})
                rebuilt=mixed_edit_candidate_matrix(graph,proposals(graph,cand),entry['step']/128,'typed_closure_compact_v2')
                err=float(np.max(abs(rebuilt-saved['x'][record_index])));feature_error=max(feature_error,err);assert err<=1e-5
                assert saved['y'][record_index]==y and saved['ids'][record_index]==tid
                record_index+=1
            final=b.restore_graph(r['final_graph']);final.validate()
            for f in names:
                assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
                assert np.array_equal(final.family_degrees(f),ref.family_degrees(f))
            counts=metrics.counts(final);assert counts['triangles']==rc['triangles']+triangle_delta
            assert record_index==32
            rows.append(dict(kind=kind,trajectory_id=tid,reference_triangles=rc['triangles'],
                final_triangles=counts['triangles'],connected=counts['connected'],**coverage(saved)))
    for kind in ['U','B']:
        z=np.load(OUT/f'train_{kind}.npz')
        for tid in range(8):
            source='U' if tid%2 else kind;q=np.load(OUT/f'cache/{source}_{tid}/data.npz')
            for k in ['x','y','ids']:assert np.array_equal(z[k][tid*32:(tid+1)*32],q[k])
    val=np.load(OUT/'validation.npz');keys=[('U',tid) for tid in range(8,12)]+[('B',8),('B',10)]
    for i,(kind,tid) in enumerate(keys):
        q=np.load(OUT/f'cache/{kind}_{tid}/data.npz')
        for k in ['x','y','ids']:assert np.array_equal(val[k][i*32:(i+1)*32],q[k])
    assert set(val['ids'])==set(range(8,12)) and checked==2304
    verify()
    put('cache_acceptance.json',dict(passed=True,independent_forward_edits=checked,inverse_labels=576,
        cpu_feature_max_error=feature_error,rows=rows,seconds=time.monotonic()-tic,
        all_relation_degrees_preserved=True,source_sha256=b.sha(__file__),
        output_hashes={str(p):b.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['cache_acceptance.json']}))
    print('cache accepted',checked,feature_error,round(time.monotonic()-tic,2),flush=True)


if __name__=='__main__':main()
