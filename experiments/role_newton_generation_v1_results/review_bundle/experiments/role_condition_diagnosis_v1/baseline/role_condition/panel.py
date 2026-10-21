"""Frozen synthetic DATA generation, not a scientific graph generator baseline."""
import itertools
import json
from pathlib import Path
import networkx as nx
import numpy as np
from hsgd.bw_preflight import common as c
from .model import extract_target,sha

TYPES=np.repeat(np.arange(4),[4,6,18,6])
PAIRS=tuple(itertools.combinations_with_replacement(range(4),2))
NAMES=['Director','Actor','Movie','Genre']

def skeleton():
    edges=[]
    for d,actors in enumerate([[0],[0,1],[1,2,3],[0,2,4,5]]):
        edges.extend((d,4+a) for a in actors)
    for m in range(18):
        edges.append((m%4,10+m))
        edges.extend((4+a,10+m) for a in [m%6,(m+1)%6])
    return set(edges)

def mg_initial(condition):
    if condition=='balanced':
        edges={(10+m,28+m%6) for m in range(18)}
        edges|={(10+m,28+(m+1)%6) for m in range(6)}
    elif condition=='extreme':
        edges={(10+m,28) for m in range(18)}
        edges|={(10+m,29 if m<2 else 28+m) for m in range(6)}
    else:raise ValueError(condition)
    return edges

def swap(edges,rng,attempts=80):
    edges=set(edges)
    for _ in range(attempts):
        pairs=sorted(edges);i,j=rng.choice(len(pairs),2,replace=False);(u,v),(x,y)=pairs[i],pairs[j]
        if u==x or v==y or (u,y) in edges or (x,v) in edges:continue
        edges.remove((u,v));edges.remove((x,y));edges.update([(u,y),(x,v)])
    return edges

def graph_case(edges,types=TYPES):
    return c.make_case(types,np.asarray(sorted(edges),dtype=np.int64).T,PAIRS)

def nx_graph(case):
    g=nx.Graph();g.add_nodes_from((i,{'type':int(t)}) for i,t in enumerate(case['space'].types))
    for (u,v),r in zip(case['clean'].T,case['space'].encode(case['clean'])[0]):g.add_edge(int(u),int(v),relation=int(r))
    return g

def isomorphic(a,b):
    return nx.is_isomorphic(nx_graph(a),nx_graph(b),node_match=lambda x,y:x['type']==y['type'],edge_match=lambda x,y:x['relation']==y['relation'])

def make_panel(seed=2026102001):
    rng=np.random.default_rng(seed);records=[];conditions={}
    for condition in ['balanced','extreme']:
        selected=[];mg=mg_initial(condition)
        for attempt in range(1000):
            mg=swap(mg,rng);case=graph_case(skeleton()|mg)
            if any(isomorphic(case,old) for old in selected):continue
            selected.append(case)
            if len(selected)==6:break
        if len(selected)!=6:raise RuntimeError('cannot obtain frozen six distinct topologies; no learned-output based selection')
        target=extract_target(selected[0]);conditions[condition]=target
        for i,case in enumerate(selected):
            d=np.bincount(np.array(sorted(mg_initial(condition))).ravel(),minlength=34)
            expected=[4]*6 if condition=='balanced' else [18,2,1,1,1,1]
            actual=np.bincount(case['clean'].ravel(),minlength=34)[28:]
            assert sorted(actual)==sorted(expected)
            mgfam=PAIRS.index((2,3));f=case['space'].encode(case['clean'])[0]
            movie_mg=np.bincount(case['clean'][:,f==mgfam].ravel(),minlength=34)[10:28]
            assert sorted(movie_mg)==[1]*12+[2]*6 and int((f==mgfam).sum())==24
            assert extract_target(case).fingerprint==target.fingerprint
            records.append(dict(id=condition+'_'+str(i),condition=condition,split='train' if i<4 else 'validation',case=case))
    counts=records[0]['case']['counts'];prior=records[0]['case']['prior']
    for r in records:
        assert np.array_equal(r['case']['counts'],counts) and np.array_equal(r['case']['prior'],prior)
    return records,conditions

def augment(case,rng):
    perm=np.arange(len(case['space'].types))
    for typ in range(case['n_types']):
        nodes=np.flatnonzero(case['space'].types==typ);perm[nodes]=rng.permutation(nodes)
    edges=np.sort(perm[case['clean']],axis=0)
    return c.make_case(case['space'].types,edges,case['space'].pairs)

def save_panel(records,targets,path):
    path=Path(path)
    payload=dict(seed=2026102001,nodes=34,type_names=NAMES,type_counts=[4,6,18,6],schema=PAIRS,
                 graphs=[dict(id=r['id'],condition=r['condition'],split=r['split'],
                              types=r['case']['space'].types.tolist(),edges=r['case']['clean'].T.tolist(),
                              family_counts=r['case']['counts'].tolist()) for r in records],
                 anonymous_targets={k:v.as_dict() for k,v in targets.items()},
                 split_evidence='all six per condition mutually nonisomorphic under true type and relation labels; augmentation after splitting',
                 MG_common_neighbors='structurally zero: no M-M/G-G/D-G/A-G support')
    path.write_text(json.dumps(payload,indent=2));return sha(payload)
