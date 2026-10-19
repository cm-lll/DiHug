import hashlib
import json
from pathlib import Path
import numpy as np
from hsgd.bw_preflight import common as c

ROOT = Path(__file__).resolve().parents[1]
BASE = 'dd53b6431303263e740abedf67e01d7dfc13d15b'
EVAL_UPDATES = [3000, 6000, 9000, 10000, 12000]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def load(name):
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    path = ROOT/cfg['data']['file']
    if c.sha(path) != cfg['data']['sha256']:
        raise ValueError('dataset hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        types, edges, rel, pairs = z['types'], z['edges'].T, z['relations']-1, z['schema']
    case = c.make_case(types, edges, tuple(map(tuple, pairs)))
    if not np.array_equal(case['space'].encode(edges)[0], rel):
        raise ValueError('relation/type mapping mismatch')
    actual = inventory(case)
    for key in ['nodes','edges','type_counts','family_counts']:
        if actual[key] != cfg['data'][key]:
            raise ValueError('dataset inventory mismatch: '+key)
    return case, cfg

def inventory(case):
    return dict(nodes=len(case['space'].types),edges=case['clean'].shape[1],
                type_counts=np.bincount(case['space'].types).tolist(),family_counts=case['counts'].tolist(),
                family_pairs=[list(x) for x in case['space'].pairs],
                slot_capacities=case['space'].counts.tolist(),all_slots=case['space'].total,
                active_slots=int(case['space'].counts[case['active']].sum()),
                deterministic_empty_slots=int(case['space'].counts[~case['active']].sum()),
                prior=case['prior'].tolist(),active=case['active'].tolist())

def degrees(case, edges=None):
    edges = case['clean'] if edges is None else np.asarray(edges, dtype=np.int64).reshape(2,-1)
    fam = case['space'].encode(edges)[0]
    d = np.zeros((len(case['space'].types),case['n_families']),dtype=np.int64)
    np.add.at(d,(edges[0],fam),1); np.add.at(d,(edges[1],fam),1)
    return d

def risk(case, T=1000):
    """Analytic forward-noise E[wedges]; NOT a reverse-chain bound."""
    types=case['space'].types; counts=np.bincount(types); d=degrees(case)
    cap=np.zeros_like(d)
    for r,(a,b) in enumerate(case['space'].pairs):
        cap[types==a,r]=counts[b]-int(a==b)
        cap[types==b,r]=counts[a]-int(a==b)
    rows=[]
    for t,a in enumerate(c.schedule(T)):
        p1=a+(1-a)*case['prior']; p0=(1-a)*case['prior']
        mu=(d*p1+(cap-d)*p0).sum(1)
        sq=(d*p1**2+(cap-d)*p0**2).sum(1)
        rows.append(dict(t=t,expected_wedges=float(((mu**2-sq)/2).sum())))
    return dict(reference_wedges=int(((d.sum(1)*(d.sum(1)-1))//2).sum()),
                forward_curve=rows,forward_max=max(x['expected_wedges'] for x in rows),
                guard=2000000,warning='Reverse states can exceed the guard; stop without truncating CN.')
