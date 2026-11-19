"""Exact CPU orbit-posterior illustration, not a production reverse sampler."""
import csv,json,math,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/identity_structure_audit_20261109'
start=time.time()
y=np.array([1.,1.,0.,0.]);ys=np.array([y,1-y])
states=((np.arange(16)[:,None]>>np.arange(4))&1).astype(float)
rows=[]
for t in [1,500,1000]:
    a=math.cos((t/1000+.008)/1.008*math.pi/2)**2/math.cos(.008/1.008*math.pi/2)**2 if t<1000 else 0.
    probs=a*ys+(1-a)*.5
    likelihood=np.prod(np.where(states[:,None,:]==1,probs[None,:,:],1-probs[None,:,:]),axis=2)
    xtlaw=likelihood.mean(1);post=likelihood/likelihood.sum(1,keepdims=True)
    marg=post@ys
    output_law=np.prod(np.where(states[None,:,:]==1,marg[:,None,:],1-marg[:,None,:]),axis=2)
    valid=output_law[:,3]+output_law[:,12]
    assert np.allclose(xtlaw.sum(),1) and np.allclose(output_law.sum(1),1)
    entropy=-sum(float((xtlaw*np.where(post[:,j]>0,post[:,j]*np.log(np.maximum(post[:,j],1e-300)),0)).sum()) for j in range(2))
    rows.append(dict(t=t,retention=a,exact_joint_posterior_valid_star_probability=1.,
        independent_clean_edge_draw_valid_star_probability=float(xtlaw@valid),
        Bayes_optimal_mean_edge_CE=entropy,
        note='Uniform distribution over two type-preserving reference permutations; exact Xt enumeration; conditional clean-graph draw, not production T1000 reverse chain'))
with (OUT/'toy_posterior.csv').open('w') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'toy_posterior_receipt.json').write_text(json.dumps(dict(status='complete',input_states=16,output_states=16,times=3,model_calls=0,seconds=time.time()-start),indent=2))
print(rows)
