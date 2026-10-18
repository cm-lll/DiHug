import numpy as np
from hsgd.bw_preflight import common as c
from role_degree.role_statistics import joint_profile
from dynamic_role.core import MatchingGuide,Layout,freeze
from hsgd.kernel import MarginalKernel

REFERENCE='0'*64 # Explicit toy engineering reference, not IMDB provenance.

def setup():
    case=c.make_case(np.array([0,0,1,2]),np.array([[0,0],[2,3]]),((0,1),(0,2)))
    l=Layout.canonical(case['space'].types,case['space'].pairs,case['prior'])
    profile=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    k=MarginalKernel(l.prior,[1.,.8,.4,.1,0.]);g=MatchingGuide(l,freeze(profile,l,k.alpha,REFERENCE))
    p=np.array([.17,.63,.46,.81]);state=np.array([1,0,0,1],np.uint8)
    return case,l,profile,k,g,p,state

def chain_setup(cn=False):
    from hsgd.fusion_checks import make
    case,l,pf,k,g,p,state=setup();model,_=make(case,cn);model.eval()
    return case,model,k,g,state
