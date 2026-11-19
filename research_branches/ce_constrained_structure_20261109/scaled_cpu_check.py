"""Single predetermined scale correction; preserve the unscaled CPU results."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cpu_check import Panel,CONFIG,OUT,DirectScore,write_csv
from scaled_controller import ScaledController

protocol=dict(version='fixed_gradient_scale_preobservation_v2',
    purpose='Correct objective-value normalization versus gradient-scale mismatch, not search a multiplier.',
    frozen_rule='lambda_struct=norm(exact initial CE gradient)/norm(exact initial structural gradient)',
    production_rule='median of 8 prespecified per-input norm ratios; requires new production calibration, no generated outcomes',
    production_times=[100,300,600,900],production_states_per_time=2,
    primal_rate=.02,dual_rate=.05,multiplier_start=1.,iterations=4000,
    observation='pre-update CE, committed after successful primal step',
    gate='finite, gradient correctness, exact controller resume; quality reported separately',
    limitations='CPU exact SGD toy only; production AdamW/EMA and stochastic variance untested.')
(OUT/'scaled_cpu_protocol.json').write_text(json.dumps(protocol,indent=2))
began=time.monotonic();results=[];curves=[];tests=[]
for name in ['triangle_tail','star']:
    panel=Panel(name);parts=[]
    for bits in panel.bits:
        m=DirectScore(panel.types,panel.q,panel.f,panel.y,bits,3).metrics
        parts.append([m.base_w1,m.base_tv,abs(m.triangles-m.ref_triangles)])
    parts=np.array(parts)
    for seed in CONFIG['initialization_seeds']:
        theta=np.random.default_rng(seed).normal(0,.4,size=7)
        ce0,s0,cg,sg=panel.evaluate(theta)
        assert np.linalg.norm(cg)>0 and np.linalg.norm(sg)>0
        coefficient=float(np.linalg.norm(cg)/np.linalg.norm(sg))
        c=ScaledController(ce0,coefficient,protocol['dual_rate'])
        assert abs(np.linalg.norm(coefficient*sg)/np.linalg.norm(cg)-1)<1e-12
        max_violation=0.
        for k in range(1,protocol['iterations']+1):
            ce,s,cg,sg=panel.evaluate(theta)
            a,b=c.coefficients();theta-=protocol['primal_rate']*(a*cg+b*sg)
            c.commit_observation(ce)
            if k==2000:
                cr=ScaledController.from_state_dict(json.loads(json.dumps(c.state_dict())))
                ce1,_,cg1,sg1=panel.evaluate(theta)
                a,b=c.coefficients();ar,br=cr.coefficients()
                assert np.array_equal(theta-protocol['primal_rate']*(a*cg1+b*sg1),theta-protocol['primal_rate']*(ar*cg1+br*sg1))
                cr.commit_observation(ce1)
                cc=ScaledController.from_state_dict(c.state_dict());cc.commit_observation(ce1)
                assert cr.state_dict()==cc.state_dict()
            post=panel.evaluate(theta);max_violation=max(max_violation,post[0]/ce0-1.)
            if k%100==0:
                curves.append(dict(case=name,seed=seed,iteration=k,CE=post[0],structure=post[1],multiplier=c.multiplier,relative_violation=post[0]/ce0-1.))
        z=panel.x@theta;terms=np.logaddexp(0,z)-panel.y*z;component=[]
        for row in z:
            mass=np.exp(np.sum(panel.bits*(-np.logaddexp(0,-row))+(1-panel.bits)*(-np.logaddexp(0,row)),axis=1))
            component.append(mass@parts)
        w,tv,tri=np.mean(component,axis=0)
        a,b=c.coefficients()
        results.append(dict(case=name,seed=seed,CE_start=ce0,structure_start=s0,CE_final=post[0],structure_final=post[1],
            coefficient=coefficient,multiplier=c.multiplier,relative_violation=post[0]/ce0-1.,
            maximum_transient_relative_violation=max_violation,stationarity_L2=float(np.linalg.norm(a*post[2]+b*post[3])),
            degree_W1=float(w),joint_TV=float(tv),triangle_abs_error=float(tri),
            positive_CE=float(terms[:,panel.y==1].mean()),negative_CE=float(terms[:,panel.y==0].mean()),
            positive_slots_per_state=int(panel.y.sum()),negative_slots_per_state=int((1-panel.y).sum()),
            finite=bool(np.isfinite(theta).all()),exact_controller_restore=True))
        tests.append(dict(case=name,seed=seed,scale_identity=True,restore=True))
write_csv(OUT/'scaled_toy_results.csv',results);write_csv(OUT/'scaled_toy_curves.csv',curves)
files=[Path(__file__),Path(__file__).with_name('scaled_controller.py')]
(OUT/'scaled_audit.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-began,
    cases=6,model_calls=0,GPU_hours=0,checks=tests,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
print(json.dumps(results,indent=2))
