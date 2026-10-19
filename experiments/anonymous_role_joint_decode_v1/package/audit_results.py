"""Independent read-only finite-sum/degree audit; imports no production decoder.

Numerical checks fixed before this audit: probability atol=1e-12;
metric absolute error <=1e-10, except exp(entropy) <=1e-8. No selection.
"""
import argparse
import bisect
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topology_statistics(bits, slots, families, types, r):
    """Separate adjacency-set, Python-loop oracle (not Universe machinery)."""
    degrees=[];connected=[];triangles=[]
    for row in bits:
        adj=[set() for _ in types];d=[[0]*r for _ in types]
        for selected,(u,v),family in zip(row,slots.T,families):
            if selected:
                adj[u].add(v);adj[v].add(u);d[u][family]+=1;d[v][family]+=1
        reach={0};queue=[0]
        while queue:
            for v in adj[queue.pop()]:
                if v not in reach:reach.add(v);queue.append(v)
        connected.append(len(reach)==len(types))
        triangles.append(sum(v in adj[u] and w in adj[u] and w in adj[v]
                             for u,v,w in itertools.combinations(range(len(types)),3)))
        degrees.append(d)
    return np.array(degrees),np.array(connected),np.array(triangles)


def audit(small,imdb,reference):
    start=time.monotonic();cpu=time.process_time();checks=[];bindings={}
    def near(name,left,right,tol=1e-10):
        error=float(np.max(np.abs(np.asarray(left)-np.asarray(right))))
        if not np.isfinite(error) or error>tol:raise AssertionError((name,error,tol))
        checks.append(dict(name=name,maximum_absolute_error=error,tolerance=tol))
    summary=json.loads((small/'summary.json').read_text())
    cache={}
    for index,case in enumerate(summary['cases']):
        path=small/case['name']/'exact_tables.npz';bindings[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        bits=a['bits'];slots=a['slots'];fam=a['families'];types=a['types'];r=len(case['target']['family_pairs'])
        key=(tuple(types),tuple(fam))
        if key not in cache:cache[key]=topology_statistics(bits,slots,fam,types,r)
        degrees,conn,tri=cache[key]
        target={int(t):Counter({tuple(row):int(n) for row,n in zip(g['rows'],g['counts'])})
                for t,g in case['target']['groups'].items()}
        roles=np.array([all(Counter(tuple(d[v]) for v in np.flatnonzero(types==t))==g
                            for t,g in target.items()) for d in degrees])
        if not np.array_equal(roles,a['role_mask']):raise AssertionError('joint role mask')
        p=np.array(case['probabilities'])
        if index<3:
            if index==0:raw=np.full(len(bits),1./32768)
            else:
                wanted=p>.5;matches=np.sum(bits==wanted,axis=1)
                # Exact rational integer weights, independent of log-score implementation.
                raw=np.array([4**int(m)/5**15 for m in matches])
        else:raw=np.prod(np.where(bits,p,1-p),axis=1)
        logq=np.log(raw);near(case['name']+'/logq',logq,a['log_q'],1e-12)
        near(case['name']+'/q(C)',raw[roles].sum(),case['truth']['q_C'],1e-12)
        expected=dict(independent=raw,uniform_feasible=roles/roles.sum(),
                      conditional_T1=np.where(roles,raw,0)/raw[roles].sum(),
                      conditional_T2=np.where(roles,np.sqrt(raw),0)/np.sqrt(raw[roles]).sum())
        mapcode=int(np.argmax(np.where(roles,raw,-1)))
        expected['MAP']=np.eye(1,len(bits),mapcode,dtype=float).ravel()
        rng=np.random.default_rng(np.random.SeedSequence([2026101907,index]))
        near(case['name']+'/uniform stream',a['shared_uniforms'],rng.random(4096),0)
        for method,w in expected.items():
            saved=a[method+'_probabilities'];row=case['methods'][method]
            near(case['name']+'/'+method+'/weights',saved,w,1e-12)
            cdf=np.cumsum(saved).tolist();cdf[-1]=1.
            draws=[bisect.bisect_right(cdf,float(u)) for u in a['shared_uniforms']]
            if draws!=a[method+'_draw_codes'].tolist():raise AssertionError('draw replay')
            if len(set(draws))!=row['sampled_distinct_labelled_graphs']:raise AssertionError('distinct draws')
            positive=w>0;h=-float(np.dot(w[positive],np.log(w[positive])))
            marg=w@bits
            metrics=dict(exact_joint_role_rate=float(w@roles),entropy_nats=h,
                effective_graph_count_exp_entropy=np.exp(h),inverse_simpson_graph_count=1/float(w@w),
                maximum_single_graph_probability=float(w.max()),
                expected_two_graph_edge_symmetric_difference=float(sum(2*x*(1-x) for x in marg)),
                connected_rate=float(w@conn),mean_triangles=float(w@tri),mean_original_log_q_nats=float(w@logq))
            for name,value in metrics.items():
                near(case['name']+'/'+method+'/'+name,value,row[name],1e-8 if 'graph_count' in name else 1e-10)
        fixed=a['fixed_assignment_bits'];d=degrees[case['fixed_assignment_graph_code']]
        if not np.array_equal(d,a['assigned_joint_degrees']):raise AssertionError('fixed matching degrees')
        fixedmask=np.all(degrees==d[None,:,:],axis=(1,2))
        near(case['name']+'/fixed MAP',float(logq[fixedmask].max()),case['fixed_assignment_MAP']['log_q'],1e-12)
    # No optimizer, reference identity scoring or additional graph is used below.
    graph=imdb/'engineering_graph.npz';summary_path=imdb/'summary.json'
    bindings[str(graph)]=sha(graph);bindings[str(summary_path)]=sha(summary_path);bindings[str(reference)]=sha(reference)
    report=json.loads(summary_path.read_text());target=json.loads((imdb/'anonymous_target.json').read_text())['profile']
    with np.load(graph,allow_pickle=False) as z:
        types=z['types'];edges=z['edges'];families=z['families'];assigned=z['assigned_joint_degrees'];actual=z['actual_joint_degrees']
    counts=Counter();degree=np.zeros_like(assigned);pairs=target['family_pairs'];seen=set()
    for (u,v),family in zip(edges.T,families):
        u=int(u);v=int(v);family=int(family)
        if not 0<=u<v<len(types) or (u,v) in seen:raise AssertionError('simple upper-triangle graph')
        seen.add((u,v))
        if sorted([int(types[u]),int(types[v])])!=pairs[family]:raise AssertionError('family schema')
        degree[u,family]+=1;degree[v,family]+=1;counts[family]+=1
    if not np.array_equal(degree,assigned) or not np.array_equal(degree,actual):raise AssertionError('IMDB assigned degrees')
    for t,g in target['groups'].items():
        if Counter(tuple(row) for row in degree[types==int(t)])!=Counter({tuple(row):n for row,n in zip(g['rows'],g['counts'])}):raise AssertionError('IMDB joint multiset')
    with np.load(reference,allow_pickle=False) as z:
        reference_degree=np.zeros_like(degree)
        for (u,v),f in zip(z['edges'],z['relations']-1):reference_degree[u,f]+=1;reference_degree[v,f]+=1
        if not np.array_equal(types,z['types']):raise AssertionError('type inventory')
    for t in np.unique(types):
        if Counter(map(tuple,degree[types==t]))!=Counter(map(tuple,reference_degree[types==t])):raise AssertionError('reference degree profile')
    if [counts[f] for f in range(len(pairs))]!=report['relation_edges']:raise AssertionError('relation counts')
    if sha(graph)!=report['graph_sha256'] or sha(reference)!=report['reference_sha256']:raise AssertionError('source hash')
    return dict(status='passed',independent_implementation=True,production_module_imports=0,
        scope='all 4 enumerated case probability tables, exact metrics, draw replay, fixed MAP; IMDB exported graph degrees/schema/hash only',
        topology_classification_independent_reaudit=False,optimizer_dual_bounds_independent_reaudit=False,
        checks=checks,IMDB_exact_degree_and_joint_multiset=True,IMDB_relation_counts=[counts[f] for f in range(len(pairs))],
        input_hashes=bindings,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--small',type=Path,required=True);p.add_argument('--imdb',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    result=audit(a.small,a.imdb,a.reference)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','input_hashes')}))
