"""Unchanged CE plus hard-graph rooted connection score, unbiased edge subsampling."""
import sys,copy,json,hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from score import RootedScore
from gpu_score import GPUIncremental
VERSION='rooted_connection_v1_HT1024'
PROPOSED_WEIGHT=.1
EDGE_SAMPLE=1024

class Objective:
    def __init__(self,case,cfg,dataset,kind):
        if kind!='RC':raise ValueError('unregistered objective')
        types=case['space'].types;n=len(types);schema=case['space'].pairs;prior=case['prior']
        if np.any(np.asarray(prior)==1):raise ValueError('this registered panel has no fixed-one relation; not silently dropping fixed edges')
        u,v=np.triu_indices(n,1);lookup={tuple(pair):f for f,pair in enumerate(schema)}
        families=np.array([lookup.get(tuple(sorted((int(types[i]),int(types[j])))),-1) for i,j in zip(u,v)])
        mutable=np.array([f>=0 and 0<prior[f]<1 for f in families]);q=np.stack([u[mutable],v[mutable]]);f=families[mutable]
        a=np.zeros((n,n));i,j=case['clean'];a[i,j]=a[j,i]=1
        labels=a[q[0],q[1]];self.keys=q[0]*n+q[1];self.n=n;self.f=f
        self.core=RootedScore(types,q,f,labels,len(schema));self.rng=np.random.default_rng(np.random.SeedSequence([int(cfg['seed']),2026110901]))
        self.calls=0;self.last={}
        self.binding=dict(version=VERSION,data_sha256=cfg['datasets'][dataset]['sha256'],schema=[list(map(int,p)) for p in schema],prior=list(map(float,prior)),
            types_sha256=hashlib.sha256(np.asarray(types,dtype=np.int64).tobytes()).hexdigest(),query_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            target_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),score=self.core.version,weight=PROPOSED_WEIGHT,
            gradient='one hard graph; uniform without replacement min(1024,M) edges; Horvitz Thompson M/m exact conditional differences',
            scale='reference log1p RMS; equal active feature blocks',CE='unaltered sumCE/full schema',R=1,T=1000)
    def objective_binding(self):return copy.deepcopy(self.binding)
    def assert_binding(self,b):
        if json.dumps(b,sort_keys=True)!=json.dumps(self.binding,sort_keys=True):raise ValueError('objective binding mismatch')
    def state_dict(self):return dict(version=VERSION,rng=copy.deepcopy(self.rng.bit_generator.state),calls=self.calls)
    def load_state_dict(self,z):
        if z['version']!=VERSION:raise ValueError('auxiliary state version')
        self.rng.bit_generator.state=copy.deepcopy(z['rng']);self.calls=int(z['calls'])
    @contextmanager
    def transaction(self):
        old=self.state_dict()
        try:yield
        except BaseException:self.load_state_dict(old);raise
    def weights(self,query,families,inp,*,device,dtype):
        self.last=dict(weight_min=1.,weight_max=1.,weight_mean_error=0.,objective_kind='RC')
        return torch.ones(len(families),device=device,dtype=dtype)
    def __call__(self,p,query,families,inp,*,logits):
        keys=np.minimum(*query)*self.n+np.maximum(*query);order=np.argsort(keys)
        if not np.array_equal(keys[order],self.keys) or not np.array_equal(np.asarray(families)[order],self.f):raise ValueError('complete mutable support/query family binding')
        if logits.shape!=(len(p),2):raise ValueError('binary clean-state logit order required')
        prob=p.detach().double().cpu().numpy()[order]
        if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):raise FloatingPointError('bad clean edge probabilities')
        bits=(self.rng.random(len(prob))<prob).astype(float)
        ix=np.sort(self.rng.choice(len(prob),size=min(EDGE_SAMPLE,len(prob)),replace=False))
        inc=GPUIncremental(self.core,bits,str(p.device));difference=inc.differences(ix)
        derivative=prob[ix]*(1-prob[ix])*difference*(len(prob)/len(ix))
        index=torch.as_tensor(order[ix],device=p.device);margin=logits[index,1]-logits[index,0]
        surrogate=(margin*torch.as_tensor(derivative,device=p.device,dtype=margin.dtype)).sum()
        auxiliary=surrogate-surrogate.detach()+surrogate.new_tensor(inc.base_score)
        self.calls+=1
        self.last.update(auxiliary_calls=self.calls,objective_value=inc.base_score,hard_edges=int(bits.sum()),auxiliary_sampled_slots=len(ix),auxiliary_total_slots=len(prob),
            auxiliary_bits_sha256=hashlib.sha256(bits.astype(np.uint8).tobytes()).hexdigest(),auxiliary_selected_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
            conditional_difference_absmax=float(np.max(abs(difference))),HT_logit_gradient_l2=float(np.linalg.norm(derivative)))
        return auxiliary,p.new_tensor([inc.base_score])

def make_objective(case,cfg,dataset,kind):return Objective(case,cfg,dataset,kind)
def checkpoint_with_aux(boundary,binding,objective):
    z=boundary.checkpoint(binding);z['auxiliary_state']=objective.state_dict()
    if objective.calls!=boundary.completed:raise ValueError('auxiliary/main commit counters disagree')
    return z
