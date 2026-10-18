"""Reproducible numeric witnesses; small probability/engineering fixtures only."""
import copy
import itertools
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_stepwise import setup_small,case_guide
from stepwise_role.core import freeze,StepwiseGuide
from stepwise_role.sampling import reverse_step


def main():
    start,cpu=time.perf_counter(),time.process_time()
    _,l,pf,k,g,state=setup_small();p=np.array([.17,.63,.46,.81])
    gradients=[];worst=0.
    for t in (1,2,3,4):
        q,loss,grad,_,affine=g.posterior_gradient(p,state,k,t)
        z=np.log(p)-np.log1p(-p);numeric=[]
        for i in range(4):
            zz=z.copy();zz[i]+=1e-5
            plus=g.statistic(k.reverse(1/(1+np.exp(-zz)),state,l.families,t-1,t),t-1)[0]
            zz[i]-=2e-5
            minus=g.statistic(k.reverse(1/(1+np.exp(-zz)),state,l.families,t-1,t),t-1)[0]
            numeric.append((plus-minus)/2e-5)
        error=float(np.max(abs(grad-numeric)));worst=max(worst,error)
        _,_,record=g.apply(p,state,k,t)
        gradients.append(dict(t=t,gradient_finite_difference_max_abs=error,posterior_affine_error=affine,guidance=record))
    other=copy.deepcopy(pf);other['groups']['0']['rows']=[[0,1],[1,0]];other['groups']['0']['counts']=[1,1]
    gg=StepwiseGuide(l,freeze(other,l,k.alpha))
    _,a,ra=g.apply(p,state,k,1);_,b,rb=gg.apply(p,state,k,1)
    _,_,zero=g.apply(np.full(4,.5),state,k,1)
    forward_max=0.
    clean=np.array([1,0,1,0])
    for s in range(5):
        pp=k.alpha[s]*clean+(1-k.alpha[s])*l.prior[l.families]
        for c0,w in g.omega.items():
            cf=np.zeros(32,dtype=complex)
            for bits in itertools.product((0,1),repeat=4):
                bits=np.array(bits);mass=np.prod(np.where(bits,pp,1-pp))
                deg=np.zeros((len(l.types),len(l.pairs)),int)
                for e,(u,v) in enumerate(l.slots.T):
                    deg[u,l.families[e]]+=bits[e];deg[v,l.families[e]]+=bits[e]
                cf+=mass*np.exp(1j*deg[l.types==int(c0)]@w.T).mean(0)
            forward_max=max(forward_max,float(abs(cf-g.targets[c0][s]).max()))
    expected_discrete_loss=0.
    for bits in itertools.product((0,1),repeat=4):
        bits=np.array(bits);expected_discrete_loss+=float(np.prod(np.where(bits,p,1-p)))*g.statistic(bits,0)[0]
    chains=[]
    for cn in (False,True):
        case,model,kk,guide,x=case_guide(cn);rng=np.random.default_rng(1789)
        steps=[]
        for t in range(4,0,-1):
            x,report=reverse_step(model,case,x,kk,t,20,1789,rng,'cpu',arm='D' if cn else 'C',guide=guide)
            steps.append(report)
        chains.append(dict(arm='D' if cn else 'C',steps=steps,neural_backward=False,optimizer_updates=0,
                           scope='untrained small CPU engineering chain, NOT quality evidence'))
    result=dict(status='passed_engineering_witnesses',gradient_max_abs=worst,
                exhaustive_forward_target_max_abs=forward_max,
                proxy_vs_expected_discrete_loss=dict(proxy=g.statistic(p,0)[0],expected_discrete=expected_discrete_loss),
                different_joint_targets_q_max_abs=float(abs(a-b).max()),same_marginal_target_responses=[ra,rb],
                positive_loss_zero_gradient=zero,small_probability_steps=gradients,engineering_chains=chains,
                wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                training_updates=0,scientific_graphs=0,GPU_use=0)
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('engineering_chains','small_probability_steps','same_marginal_target_responses')},indent=2))


if __name__=='__main__':main()
