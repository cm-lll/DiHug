"""Small deterministic probability witnesses only; no six real inputs."""
import sys
import json
import time
import resource
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_multiscale import example,enumerate_cf_tail


def main(output):
    started,cpu=time.perf_counter(),time.process_time()
    l,pf,k,g,p,state=example();gradient_max=0.;target_max=0.;tail_max=0.;conditional_max=0.;steps=[]
    clean=np.zeros(len(p));clean[[0,1,3,6]]=1
    for s in range(4):
        probabilities=k.alpha[s]*clean+(1-k.alpha[s])*l.prior[l.families]
        cf,tail=enumerate_cf_tail(l,g,probabilities,s);tc,tt=g.target(s)
        for c in cf:
            target_max=max(target_max,float(abs(cf[c]-tc[c]).max(initial=0)))
            tail_max=max(tail_max,float(abs(tail[c]-tt[c]).max(initial=0)))
    ec,et=enumerate_cf_tail(l,g,p,0);details=g.statistic(p,0,True)[2]
    for c in ec:
        conditional_max=max(conditional_max,float(abs(ec[c]-details['mean_cf'][c]).max(initial=0)),
                            float(abs(et[c]-details['mean_tail'][c]).max(initial=0)))
    for t in (1,2,3):
        q,loss,grad,_,affine=g.posterior_gradient(p,state,k,t)
        numeric=np.zeros_like(p)
        for i in np.flatnonzero(~l.fixed0):
            z=np.log(p[i])-np.log1p(-p[i]);a=p.copy();b=p.copy()
            a[i]=1/(1+np.exp(-z-1e-5));b[i]=1/(1+np.exp(-z+1e-5))
            numeric[i]=(g.statistic(k.reverse(a,state,l.families,t-1,t),t-1)[0]-g.statistic(k.reverse(b,state,l.families,t-1,t),t-1)[0])/2e-5
        error=float(abs(grad-numeric).max());gradient_max=max(gradient_max,error)
        _,_,record=g.apply(p,state,k,t)
        steps.append(dict(t=t,posterior_affine_error=affine,gradient_error=error,controller=record))
    result=dict(status='passed_small_probability_witnesses',CF_target_enumeration_max=target_max,
                tail_target_enumeration_max=tail_max,conditional_statistic_enumeration_max=conditional_max,
                posterior_gradient_finite_difference_max=gradient_max,steps=steps,
                wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                real_six_states_tested=False,model_calls=0,scientific_graphs=0,GPU_use=0)
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='steps'},indent=2))


if __name__=='__main__':main(sys.argv[1])
