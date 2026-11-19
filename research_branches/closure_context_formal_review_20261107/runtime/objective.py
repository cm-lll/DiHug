"""Relation-normalized binary frozen-Xt closure weighting of CE; original PB unchanged."""
import json,hashlib
import numpy as np
import torch
from pb_degree import RelationDegreeCDF
VERSION='binary_closure_context_relation_mean_v1'
PROPOSED_WEIGHT=1.0
class Objective:
    def __init__(self,case,cfg,dataset):
        self.pb=RelationDegreeCDF(case['space'].types,case['space'].pairs,case['prior'],case['clean'],data_sha256=cfg['datasets'][dataset]['sha256'])
        self.last={}
    def objective_binding(self):
        return dict(version=VERSION,pb=self.pb.objective_binding(),PB_weight=1.,T=1000,R=1,
            formula='sum_e [(1+hit_Xt(e))/(1+mean_relation(hit_Xt))]*supported_CE_e/S + PB',
            hit='at least one common neighbor in current frozen Xt; no label or prediction dependence',
            normalization='all unique mutable unordered slots of each relation; fixed S=n(n-1)/2',
            scope='full R1 only; no weight used at inference; no triangle count objective')
    def assert_binding(self,binding):
        if json.dumps(binding,sort_keys=True)!=json.dumps(self.objective_binding(),sort_keys=True):raise ValueError('closure-context/PB objective binding mismatch')
    def weights(self,query,families,inp,*,device,dtype):
        owner=self.pb;query=np.asarray(query);families=np.asarray(families)
        if query.shape!=(2,len(families)):raise ValueError('query shape')
        if np.any(query<0) or np.any(query>=owner.n) or np.any(query[0]==query[1]):raise ValueError('query endpoints')
        keys=np.minimum(*query)*owner.n+np.maximum(*query)
        if not np.array_equal(np.sort(keys),owner.keys):raise ValueError('full unique mutable R1 support required')
        if np.any(families<0) or np.any(families>=len(owner.schema)):raise ValueError('family range')
        if not np.array_equal(np.sort(owner.types[query].T,axis=1),np.asarray(owner.schema)[families]):raise ValueError('ordered relation semantics')
        edges=np.asarray(inp['edges'],dtype=np.int64)
        if edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('Xt edge shape')
        if np.any(edges<0) or np.any(edges>=owner.n) or np.any(edges[0]==edges[1]):raise ValueError('Xt endpoints')
        ek=np.minimum(*edges)*owner.n+np.maximum(*edges)
        if len(np.unique(ek))!=len(ek):raise ValueError('duplicate Xt edge')
        lookup={r:f for f,r in enumerate(owner.schema)}
        ef=np.array([lookup.get(tuple(sorted((int(owner.types[u]),int(owner.types[v])))),-1) for u,v in edges.T],dtype=np.int64)
        if np.any(ef<0) or np.any(owner.prior[ef]==0):raise ValueError('illegal/fixed-zero Xt edge')
        if not np.array_equal(ef,np.asarray(inp['edge_families'])):raise ValueError('Xt family mapping')
        adj=[set() for _ in range(owner.n)]
        for u,v in edges.T:adj[u].add(int(v));adj[v].add(int(u))
        hit=np.array([bool(adj[u]&adj[v]) for u,v in query.T]);weights=np.ones(len(hit),dtype=np.float64);groups=[]
        for f in np.unique(families):
            ix=families==f;weights[ix]=(1+hit[ix].astype(float))/(1+hit[ix].mean())
            groups.append(dict(relation=int(f),slots=int(ix.sum()),hit_fraction=float(hit[ix].mean()),mean_weight=float(weights[ix].mean())))
        error=max((abs(x['mean_weight']-1) for x in groups),default=0.)
        if error>1e-12 or not np.isfinite(weights).all() or weights.min()<.5 or weights.max()>2:raise FloatingPointError('weight normalization/bounds')
        self.last=dict(weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean_error=error,
            closure_hit_fraction=float(hit.mean()),relation_weights=groups,frozen_Xt_sha256=hashlib.sha256(np.sort(ek).tobytes()).hexdigest())
        return torch.as_tensor(weights,device=device,dtype=dtype)
    def __call__(self,p,query,families,inp,*,logits):
        pb,detail=self.pb(p,query,families);self.last['PB_loss']=float(pb.detach());return pb,detail
def make_objective(case,cfg,dataset):return Objective(case,cfg,dataset)

