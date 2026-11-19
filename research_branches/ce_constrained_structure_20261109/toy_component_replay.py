"""Replay the frozen CPU toy arithmetic to retain component tradeoffs.

No change of hyperparameters or selection; original tables remain untouched.
"""
import csv
import json
import time
import numpy as np
from scipy.special import expit
from cpu_check import Panel, CONFIG, OUT, DirectScore, write_csv
from controller import Controller

started=time.monotonic()
with (OUT/'toy_results.csv').open() as f:old=list(csv.DictReader(f))
out=[]
for name in ['triangle_tail','star']:
    p=Panel(name);parts=[]
    for b in p.bits:
        m=DirectScore(p.types,p.q,p.f,p.y,b,3).metrics
        parts.append([m.base_w1,m.base_tv,abs(m.triangles-m.ref_triangles)])
    parts=np.array(parts)
    for seed in CONFIG['initialization_seeds']:
        initial=np.random.default_rng(seed).normal(0,.4,size=7)
        ce0,s0,_,_=p.evaluate(initial)
        for arm in ['fixed_joint','constrained']:
            theta=initial.copy();c=Controller(ce0,s0,CONFIG['dual_rate'])
            for it in range(CONFIG['iterations']):
                ce,s,cg,sg=p.evaluate(theta)
                if arm=='fixed_joint':grad=cg+CONFIG['fixed_joint_lambda']*sg
                else:
                    a,b=c.coefficients();grad=a*cg+b*sg
                theta-=CONFIG['primal_rate']*grad
                post=p.evaluate(theta)
                if arm=='constrained':c.commit(post[0])
            previous=next(r for r in old if r['case']==name and int(r['seed'])==seed and r['arm']==arm)
            assert post[0]==float(previous['CE_final']) and post[1]==float(previous['structure_final'])
            z=p.x@theta;terms=np.logaddexp(0,z)-p.y*z
            components=[]
            for row in z:
                mass=np.exp(np.sum(p.bits*(-np.logaddexp(0,-row))+(1-p.bits)*(-np.logaddexp(0,row)),axis=1))
                components.append(mass@parts)
            w,tv,tri=np.mean(components,axis=0)
            out.append(dict(case=name,seed=seed,arm=arm,degree_W1=float(w),joint_TV=float(tv),triangle_abs_error=float(tri),
                positive_CE=float(terms[:,p.y==1].mean()),negative_CE=float(terms[:,p.y==0].mean()),
                positive_slots_per_state=int(p.y.sum()),negative_slots_per_state=int((1-p.y).sum()),
                exact_replay=True))
write_csv(OUT/'toy_components.csv',out)
(OUT/'component_replay_audit.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,
    comparisons=12,exact_replay=True,new_model_calls=0,GPU_hours=0,
    scope='Repeated CPU scalar toy arithmetic; no production model or graph generation.'),indent=2))
print(json.dumps(out,indent=2))
