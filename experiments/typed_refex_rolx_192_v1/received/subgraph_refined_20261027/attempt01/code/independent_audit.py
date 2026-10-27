#!/usr/bin/env python3
"""Independent CPU-only full-induced and sparse motif audit; no evaluator imports.

CLI: python independent_audit.py --graph IMDB:/path/graph.npz --graph PubMed:/path/graph.npz
Optionally --metrics normalized.json where each dataset maps to its final_induced
record. audit_one() is usable from another driver without invoking the evaluator.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, itertools, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

DEFAULT_ROOT=Path('/workspace/shared/subgraph_refined_20261027')
DEFAULT_SOURCE=Path('/workspace/shared/subgraph_preview_20261027')
QUOTAS={'IMDB':[48,56,78,10],'PubMed':[40,74,66,12]}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_graph(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def ids_for(g,path):
    for key in ('original_ids','selected_original_ids','original_node_ids'):
        if key in g:return np.asarray(g[key],dtype=np.int64)
    for name in ('original_node_ids.npy','selected_original_ids.npy'):
        p=path.parent/name
        if p.exists():return np.asarray(np.load(p,allow_pickle=False),dtype=np.int64)
    m=path.parent/'metadata.json'
    if m.exists():
        data=json.loads(m.read_text())
        for key in ('selected_original_ids','selected_nodes_original'):
            if key in data:return np.asarray(data[key],dtype=np.int64)
    raise ValueError('No explicit original-node mapping supplied')

def measure(g,typed_mp2=True,typed_mp3=False):
    t=np.asarray(g['types']);e=np.asarray(g['edges'],dtype=np.int64);r=np.asarray(g['relations'],dtype=np.int64);n=len(t)
    rows=np.concatenate((e[:,0],e[:,1]));cols=np.concatenate((e[:,1],e[:,0]))
    a=csr_matrix((np.ones(2*len(e),dtype=np.int64),(rows,cols)),shape=(n,n))
    deg=np.asarray(a.sum(axis=1)).ravel()
    closed=(a@a).multiply(a)
    tri_node=np.asarray(closed.sum(axis=1)).ravel()//2
    tri=int(tri_node.sum()//3)
    cl=np.divide(2*tri_node,deg*(deg-1),out=np.zeros(n,float),where=deg>1)
    adj=[set() for _ in range(n)];rel={}
    degree=np.zeros((n,10),dtype=np.int64)
    for (u,v),f in zip(e,r):
        u,v,f=int(u),int(v),int(f)
        adj[u].add(v);adj[v].add(u);rel[u,v]=rel[v,u]=f-1
        degree[u,f-1]+=1;degree[v,f-1]+=1
    typed_tri=Counter()
    for u in range(n):
        for v in adj[u]:
            if v<=u:continue
            for w in adj[u].intersection(adj[v]):
                if w>v:typed_tri[','.join(map(str,sorted((int(t[u]),int(t[v]),int(t[w])))))]+=1
    all_tris={','.join(map(str,k)):int(typed_tri[','.join(map(str,k))]) for k in itertools.combinations_with_replacement(range(4),3)}
    mp2=Counter()
    if typed_mp2:
        for u in range(n):
            for v,w in itertools.combinations(sorted(adj[u]),2):
                key=(int(t[v]),rel[v,u],int(t[u]),rel[u,w],int(t[w]))
                mp2[min(key,key[::-1])]+=1
    mp3=Counter()
    if typed_mp3:
        for u,v in e:
            u,v=int(u),int(v)
            for x in adj[u]-{v}:
                for y in adj[v]-{u,x}:
                    key=(int(t[x]),rel[x,u],int(t[u]),rel[u,v],int(t[v]),rel[v,y],int(t[y]))
                    mp3[min(key,key[::-1])]+=1
    mp2_total=int(np.sum(deg*(deg-1)//2))
    mp3_total=int(np.sum((deg[e[:,0]]-1)*(deg[e[:,1]]-1))-3*tri)
    return dict(nodes=n,edges=len(e),type_counts=np.bincount(t,minlength=4).tolist(),family_edges=np.bincount(r-1,minlength=10).tolist(),isolates=int(sum(deg==0)),triangles=tri,typed_triangles=all_tris,average_clustering=float(cl.mean()),mp2_total=mp2_total,mp3_total=mp3_total,mp2_counts={','.join(map(str,k)):int(v) for k,v in sorted(mp2.items())},mp3_counts={','.join(map(str,k)):int(v) for k,v in sorted(mp3.items())},total_degree=deg.tolist(),type_relation_degree_vectors=degree.tolist(),triangle_incidence=tri_node.tolist(),clustering=cl.tolist(),internal_consistency=dict(sparse_vs_enumerated_triangles=tri==sum(typed_tri.values()),sparse_vs_enumerated_mp2=not typed_mp2 or mp2_total==sum(mp2.values()),identity_vs_enumerated_mp3=not typed_mp3 or mp3_total==sum(mp3.values())))

def compare_metrics(obs,m):
    if 'final_induced' in m:m=m['final_induced']
    pairs={
      'nodes':(obs['nodes'],m['nodes']),
      'edges':(obs['edges'],m['edges']),
      'isolates':(obs['isolates'],m['isolates']),
      'family_edges':(obs['family_edges'],m['family_edges']),
      'triangles':(obs['triangles'],m['triangles']),
      'typed_triangles':(obs['typed_triangles'],m['typed_triangles']),
      'mp2_total':(obs['mp2_total'],m['mp'][1]['total']),
      'mp3_total':(obs['mp3_total'],m['mp'][2]['total']),
      'mp2_counts':(obs['mp2_counts'],m['mp'][1]['counts']),
    }
    if obs.get('mp3_counts'):pairs['mp3_counts']=(obs['mp3_counts'],m['mp'][2]['counts'])
    checks={k:a==b for k,(a,b) in pairs.items()}
    checks['average_clustering']=bool(np.isclose(obs['average_clustering'],m['average_clustering'],rtol=0,atol=1e-12))
    checks['total_degree_sum']=bool(sum(obs['total_degree'])==2*m['edges'])
    checks['mean_degree']=bool(np.isclose(np.mean(obs['total_degree']),m['degree']['mean'],rtol=0,atol=1e-12))
    return dict(passed=all(checks.values()),checks=checks,mismatches={k:{'independent':a,'evaluator':b} for k,(a,b) in pairs.items() if a!=b})

def audit_one(ds,path,source=DEFAULT_SOURCE,metrics=None):
    path=Path(path);reference=Path(source)/'inputs/reference'/f'{ds}.npz'
    g=read_graph(path);original=read_graph(reference);ids=ids_for(g,path)
    n=len(original['types']);mapping=np.full(n,-1,dtype=np.int64)
    valid_ids=ids.ndim==1 and bool(np.all((ids>=0)&(ids<n)))
    if not valid_ids:raise ValueError(f'{ds}: original ids outside reference')
    mapping[ids]=np.arange(len(ids))
    mask=np.all(mapping[original['edges']]>=0,axis=1)
    expected_e=mapping[original['edges'][mask]];expected_r=original['relations'][mask]
    obs=measure(g,typed_mp3=True);ref=measure(original,typed_mp2=False)
    checks=dict(
      nodes_exactly_192=len(g['types'])==192 and len(ids)==192,
      original_ids_unique=len(set(map(int,ids)))==len(ids),
      type_quotas_exact=obs['type_counts']==QUOTAS[ds],
      types_match_original=np.array_equal(g['types'],original['types'][ids]),
      schema_match_original=np.array_equal(g['schema'],original['schema']),
      edges_exact_induced_in_original_order=np.array_equal(g['edges'],expected_e),
      relations_exact_induced_in_original_order=np.array_equal(g['relations'],expected_r),
      no_self_loops=bool(np.all(g['edges'][:,0]!=g['edges'][:,1])),
      no_duplicate_undirected_edges=len(set(tuple(sorted(map(int,e))) for e in g['edges']))==len(g['edges']),
      relation_types_match_schema=all(tuple(sorted(map(int,g['types'][edge])))==tuple(map(int,g['schema'][int(r)-1])) for edge,r in zip(g['edges'],g['relations'])),
      no_isolates=obs['isolates']==0,
      all_original_active_relation_families_retained=bool(np.all((np.asarray(ref['family_edges'])==0)|(np.asarray(obs['family_edges'])>0))),
      sparse_vs_enumerated_triangles=obs['internal_consistency']['sparse_vs_enumerated_triangles'],
      sparse_vs_enumerated_mp2=obs['internal_consistency']['sparse_vs_enumerated_mp2'],
      identity_vs_enumerated_mp3=obs['internal_consistency']['identity_vs_enumerated_mp3'],
    )
    missing=[k for k,v in ref['typed_triangles'].items() if v>0 and obs['typed_triangles'][k]==0]
    checks['all_positive_reference_triangle_types_retained']=not missing
    if ds=='IMDB':checks['rare_AAM_triangle_retained']=obs['typed_triangles']['1,1,2']>0
    old_metrics_path=Path(source)/'evaluation/metrics.json'
    coverage=None
    if old_metrics_path.exists():
        old=json.loads(old_metrics_path.read_text())['original'][ds]
        coverage={}
        for length in (2,3):
            original_types=set(old['mp'][length-1]['counts']);final_types=set(obs[f'mp{length}_counts'])
            coverage[f'MP{length}']=dict(reference_type_count=len(original_types),final_type_count=len(final_types),missing_types=sorted(original_types-final_types),unexpected_types=sorted(final_types-original_types))
    comparison=compare_metrics(obs,metrics) if metrics else None
    if comparison:checks['matches_full_evaluator']=comparison['passed']
    return dict(dataset=ds,graph_path=str(path),graph_sha256=sha(path),reference_path=str(reference),reference_sha256=sha(reference),passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},missing_positive_reference_triangle_types=missing,observed=obs,reference_positive_triangle_types={k:v for k,v in ref['typed_triangles'].items() if v>0},metric_comparison=comparison,metapath_coverage=coverage)

def protocol_and_trace_audit(root,source,results):
    root=Path(root);receipt=json.loads((root/'protocol_freeze_receipt.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    protocol_checks=dict(protocol_sha_matches_frozen_receipt=sha(root/'protocol.json')==receipt['protocol_sha256'],optimizer_code_sha_matches_frozen_receipt=sha(root/'code/refine.py')==receipt['code_sha256'],source_graph_hashes_match=all(sha(Path(source)/'inputs/reference'/f'{ds}.npz')==h for ds,h in protocol['source_graph_sha256'].items()))
    per_dataset={}
    for ds,item in results.items():
        folder=Path(item['graph_path']).parent;trace=json.loads((folder/'optimization_trace.json').read_text());md=json.loads((folder/'metadata.json').read_text());ids=list(map(int,np.load(folder/'baseline_original_node_ids.npy',allow_pickle=False)));types=read_graph(Path(source)/'inputs/reference'/f'{ds}.npz')['types'];selected=set(ids);score=md['baseline_score'];accepted=0;valid_transitions=True;strict_improvement=True;reason_counts=Counter();hard_rejects=0;not_improved=0
        for row in trace:
            old,new=int(row['removed_original_id']),int(row['added_original_id']);valid_transitions &= old in selected and new not in selected and int(types[old])==int(types[new])==row['type']
            if row['accepted']:
                valid_transitions &= not row['hard_failures'];strict_improvement &= row['score']<score-1e-12;selected.remove(old);selected.add(new);accepted+=1;score=row['score']
            elif row['hard_failures']:
                hard_rejects+=1;reason_counts.update(row['hard_failures'])
            else:
                not_improved+=1;reason_counts['not_improved']+=1
            strict_improvement &= abs(row['retained_score']-score)<1e-12
        finalids=set(map(int,ids_for(read_graph(item['graph_path']),Path(item['graph_path']))))
        checks=dict(proposal_budget_exact=len(trace)==protocol['proposal_budget_per_dataset'],iterations_complete=[r['iteration'] for r in trace]==list(range(1,len(trace)+1)),all_transitions_same_type_and_valid=bool(valid_transitions),strict_monotone_acceptance=bool(strict_improvement),accepted_trace_replays_final_node_set=selected==finalids,accepted_count_matches_metadata=accepted==md['accepted_swaps'],rejection_reason_counts_match_metadata=dict(reason_counts)==md['rejection_counts'],final_score_matches_metadata=abs(score-md['final_score'])<1e-12,exclusive_outcomes_sum_to_budget=accepted+hard_rejects+not_improved==len(trace))
        per_dataset[ds]=dict(passed=all(checks.values()),checks=checks,proposals=len(trace),accepted=accepted,hard_rejected=hard_rejects,not_improved=not_improved,rejection_reason_counts=dict(reason_counts),rejection_reason_count_note='A hard-rejected proposal may fail multiple gates, so per-reason counts need not sum to number of rejected proposals.',final_objective=score)
    return dict(passed=all(protocol_checks.values()) and all(x['passed'] for x in per_dataset.values()),protocol_checks=protocol_checks,datasets=per_dataset)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--graph',action='append',help='DATASET:absolute/path/graph.npz');p.add_argument('--metrics',type=Path,help='Normalized JSON mapping each dataset to final_induced evaluator record');p.add_argument('--out',type=Path)
    args=p.parse_args();metrics=json.loads(args.metrics.read_text()) if args.metrics else {}
    paths={}
    for entry in args.graph or []:
        ds,path=entry.split(':',1);paths[ds]=Path(path)
    if not paths:
        for ds in QUOTAS:
            candidates=sorted((args.root/'samples'/ds).rglob('graph.npz'))
            if len(candidates)!=1:raise RuntimeError(f'{ds}: expected exactly one final graph in samples, found {len(candidates)}; supply --graph')
            paths[ds]=candidates[0]
    if set(paths)!=set(QUOTAS):raise RuntimeError('Both IMDB and PubMed final graphs required')
    results={ds:audit_one(ds,path,args.source,metrics.get(ds)) for ds,path in paths.items()}
    trace_audit=protocol_and_trace_audit(args.root,args.source,results)
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),method='Independent SciPy sparse A^2 elementwise A triangles; independent explicit neighbor-pair MP2; edge-degree identity minus 3*triangles for simple MP3 totals, independently checked with explicit middle-edge neighbor-pair enumeration for MP3 type counts; no evaluator imports.',scope='CPU-only graph inspection; no model imports, GPU use, or training',evaluator_comparison_requested=bool(args.metrics),all_passed=all(x['passed'] for x in results.values()) and trace_audit['passed'],results=results,protocol_and_trace_audit=trace_audit)
    out=args.out or args.root/'independent_audit.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({ds:dict(passed=x['passed'],checks=x['checks'],triangles=x['observed']['triangles'],mp2=x['observed']['mp2_total'],mp3=x['observed']['mp3_total']) for ds,x in results.items()},indent=2))
    return 0 if report['all_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
