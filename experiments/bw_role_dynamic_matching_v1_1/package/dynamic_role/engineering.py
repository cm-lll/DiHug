"""Explicit engineering demonstrations, NOT the six historical frozen states."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import scipy
from .core import Layout,MatchingGuide,freeze
from role_degree.role_statistics import joint_profile
from hsgd.kernel import MarginalKernel


def toy():
    # Reference: six nodes, one degree-5 hub and five degree-1 roles.
    # Input: fixed heterogeneous probabilities, not a sampled scientific graph.
    l=Layout.canonical(np.zeros(6,int),[(0,0)],[1/3])
    edges=l.slots[:,l.slots[0]==0]
    pf=joint_profile(l.types,edges,np.zeros(5,int),l.pairs)
    g=MatchingGuide(l,freeze(pf,l,[1.,.7,0.],'0'*64));k=MarginalKernel(l.prior,g.alpha)
    p=np.array([.23 if 0 in e else .11 for e in l.slots.T]);state=np.zeros(len(p),np.uint8)
    before=g.expected_degrees(p)[:,0];pp,q,report=g.apply(p,state,k,1)
    # Independent fixed engineering Monte Carlo to show nonzero sampling noise.
    # Shared uniforms solely for descriptive pairing; not used in the controller.
    rng=np.random.default_rng(2026101901);u=rng.random((4096,len(p)))
    output={}
    for label,prob in [('before',p),('after',pp)]:
        draws=u<prob;degrees=np.zeros((len(u),6),int)
        for e,(v,w) in enumerate(l.slots.T):degrees[:,v]+=draws[:,e];degrees[:,w]+=draws[:,e]
        expected=g.expected_degrees(prob)[:,0];variance=np.zeros(6)
        for e,(v,w) in enumerate(l.slots.T):variance[[v,w]]+=prob[e]*(1-prob[e])
        sorted_error=np.mean((np.sort(degrees,axis=1)-np.array([1,1,1,1,1,5]))**2,axis=1)
        output[label]=dict(expected_degrees=expected.tolist(),empirical_mean=degrees.mean(0).tolist(),
            analytic_sd=np.sqrt(variance).tolist(),empirical_sd=degrees.std(0).tolist(),
            mean_sorted_discrete_role_squared_error=float(sorted_error.mean()),
            discrete_error_P10_P90=np.quantile(sorted_error,[.1,.9]).tolist(),
            exactly_target_sorted_degree_fraction=float(np.mean(np.all(np.sort(degrees,axis=1)==[1,1,1,1,1,5],axis=1))))
    return dict(label='SIX-NODE ENGINEERING TOY ONLY',draws=4096,seed=2026101901,
        reference_role_counts=[dict(vector=[1],count=5),dict(vector=[5],count=1)],
        probability_before=p.tolist(),probability_after=pp.tolist(),guidance=report,**output,
        warning='expected-degree matching is not sampled-degree-distribution matching; tied role selection can break symmetry')


def reference_cost():
    from hsgd.bw_preflight import common as c
    case=c.imdb_case(c.PACKAGE/'data/IMDB_movie_reference.npz',c.config());l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    pf=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    start=time.perf_counter();g=MatchingGuide(l,freeze(pf,l,c.schedule(1000),c.config()['imdb_sha256']))
    prepare=time.perf_counter()-start
    # Analytic reference-derived test probabilities ONLY; neither p0 from a
    # trained model nor one of the six real frozen states. Never quality evidence.
    p=np.repeat(case['prior'],case['space'].counts)*.2;p[case['clean_ids']]+=.8
    state=np.zeros(len(p),np.uint8);state[case['clean_ids']]=1
    start=time.perf_counter();pp,q,r=g.apply(p,state,MarginalKernel(l.prior,g.alpha),20)
    return dict(label='IMDB-size engineering probabilities .8*reference_indicator+.2*prior; NOT REAL SIX STATES',
        nodes=len(l.types),slots=len(p),data_sha256=c.config()['imdb_sha256'],prepare_seconds=prepare,
        total_guide_seconds=time.perf_counter()-start,report=r,
        no_probability_collection_performed=True,no_quality_claim=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();path=Path(a.output)
    if path.exists():raise FileExistsError('engineering output must be fresh')
    start=time.perf_counter();cpu=time.process_time();result=dict(toy=toy(),reference_size_cost=reference_cost())
    import torch,torch_geometric,pytest
    result.update(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        torch=torch.__version__,pyg=torch_geometric.__version__,pytest=pytest.__version__),
        wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_calls=0,GPU_use=0,training_updates=0,new_scientific_graphs=0,
        engineering_bernoulli_draws=8192)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('toy','reference_size_cost')},indent=2))


if __name__=='__main__':main()
