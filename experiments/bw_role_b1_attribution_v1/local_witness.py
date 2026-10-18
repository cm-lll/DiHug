"""Deterministic SMALL engineering witnesses; never uses the notebook B1."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent/'package';sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'b1_tests')]
import numpy as np
from test_multiscale import example
from test_attribution import scalar_fd,genre_example,four_type_fixture
from hsgd.kernel import MarginalKernel
from role_degree.role_statistics import joint_profile
from multiscale_role.core import MultiScaleGuide,freeze
from b1_attribution.math import components,posterior_chain,sum_certificate,replay,independent_cf_gradient,genre_laws
from multiscale_role.evaluation import genre64,degree_w1


def main(path):
    maximum=0.;certs=[]
    for t in (1,2):
        l,pf,k,g,p,state=example();q,factor,_=posterior_chain(p,state,k,g,t)
        info,parts=components(q,g,t-1,True,factor);native=g.posterior_gradient(p,state,k,t)[2]
        certs.append(sum_certificate(parts,native)[0])
        for key,grad in parts.items():
            f=lambda x:components(k.reverse(x,state,l.families,t-1,t),g,t-1)[0]['losses'][key]
            fd=scalar_fd(f,p,(p>0)&(p<1)&~l.fixed0&~l.fixed1);maximum=max(maximum,float(abs(fd-grad).max()))
    case,l,b,col,cp=four_type_fixture();pf=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    k=MarginalKernel(l.prior,b['alpha']);g=MultiScaleGuide(l,freeze(pf,l,k.alpha));q,factor,_=posterior_chain(b['p0'],b['state'],k,g,1)
    info,parts=components(q,g,0,True,factor);certs.append(sum_certificate(parts,g.posterior_gradient(b['p0'],b['state'],k,1)[2])[0])
    l,pf,k,g,p,state=example();q,factor,_=posterior_chain(p,state,k,g,1)
    w=np.zeros((64,len(l.pairs)));w[:,1]=np.linspace(.001,.1,64);spec=dict(Genre_type=1,omega64=w.tolist(),CF_normalizer=1)
    value,grad,_=independent_cf_gradient(q,g,0,spec)
    fd=scalar_fd(lambda x:genre64(k.reverse(x,state,l.families,0,1),g,0,spec)['error'],p,(p>0)&(p<1)&~l.fixed0)
    before,after,record=g.apply(p,state,k,1) # tiny fixture creation ONLY
    pp,qq,dz,repro=replay(p,state,k,g,1,record,g.posterior_gradient(p,state,k,1)[2])
    ll,gg,q0=genre_example();law,arrays=genre_laws(q0,q0*.9,gg,0,genre='1')
    expected=degree_w1(q0,gg,0,only_types={'1'})['1'][0]['W1']
    result=dict(scope='SMALL deterministic CPU fixtures, NOT real B1/IMDB observations',gradient_sum=certs,
        component_logit_gradient_finite_difference_max_abs_error=maximum,
        independent_metric_logit_gradient_finite_difference_max_abs_error=float(abs(grad*factor-fd).max()),
        old_chosen_update_p_exact=np.array_equal(before,pp),old_chosen_update_q_exact=np.array_equal(after,qq),
        W1_interval_sum_error=law['states']['before']['W1_additivity_error'],
        full_PMF_W1_existing_metric_error=abs(law['states']['before']['W1']-expected),
        training=0,scientific_graphs=0,real_B1_executed=False,GPU_use=0)
    Path(path).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
