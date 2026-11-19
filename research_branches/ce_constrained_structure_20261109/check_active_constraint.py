"""Known-solution constraint and checkpoint tests; CPU scalars only."""
import json
from pathlib import Path
import numpy as np
from controller import Controller

OUT=Path(__file__).resolve().parents[2]/'results/ce_constrained_structure_20261109'

def run(target,steps=4000,interrupt=False):
    # CE=x^2+0.1 <=0.35; normalized structure=(x-target)^2.
    # Start at x=0.5, the feasible boundary; active optimum for target=1 is x=0.5.
    x=.5
    c=Controller(.35,1.,dual_rate=.05)
    history=[]
    for k in range(steps):
        old_ce=x*x+.1
        # Simultaneous primal-descent/dual-ascent uses PRE-update CE, so no
        # extra post-update model forward is implied by this streaming variant.
        x-=.02*(2*(x-target)+c.multiplier*2*x/.35)
        c.commit(old_ce)
        if interrupt and k==1999:
            payload=json.loads(json.dumps(dict(x=x,controller=c.state_dict())))
            x=payload['x'];c=Controller.from_state_dict(payload['controller'])
        history.append((x,c.multiplier,x*x+.1-.35))
    return x,c,np.asarray(history)

rows=[]
for target,expected_x,expected_multiplier in [(1.,.5,.35),(.2,.2,0.)]:
    x,c,h=run(target)
    xr,cr,hr=run(target,interrupt=True)
    assert np.array_equal(h,hr) and c.state_dict()==cr.state_dict()
    stationarity=2*(x-target)+c.multiplier*2*x/.35
    violation=x*x+.1-.35
    assert abs(x-expected_x)<1e-7
    assert abs(c.multiplier-expected_multiplier)<1e-7
    assert abs(stationarity)<1e-7 and violation<1e-7
    rows.append(dict(target=target,x=x,multiplier=c.multiplier,CE=x*x+.1,
        CE_budget=.35,stationarity=stationarity,violation=violation,
        complementarity=c.multiplier*violation,
        maximum_transient_violation=float(h[:,2].max()),
        exact_resume=True,passed=True))
# Impossible CE budget: show the controller cannot manufacture feasibility.
c=Controller(.5,1.,dual_rate=.05)
for _ in range(100):c.commit(1.)
assert c.multiplier>5. and 1./.5-1.>0
rows.append(dict(case='infeasible_budget',minimum_CE=1.,budget=.5,
    multiplier_after_100=c.multiplier,passed=True,
    interpretation='Violation persists; growing multiplier is not successful optimization.'))
(OUT/'active_constraint_checks.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
