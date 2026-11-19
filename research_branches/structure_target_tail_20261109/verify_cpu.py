"""Independent scalar-root certificate for the two active tail constraints."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib,json,time,resource
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
start=time.monotonic();ROOT=Path(__file__).resolve().parents[2];p=ROOT/'results/structure_target_tail_20261109'
z=np.load(p/'targets.npz');mu=z['sparse_base'];loss=z['L'];bad=loss>1+1e-12;excess=np.maximum(0,loss-1)
def cond(nu):
    x=np.log(mu[bad])-nu*excess[bad];w=np.exp(x-logsumexp(x));return w
# Desired bad mass .1 and excess .1 imply bad-conditional excess 1.
nu=brentq(lambda v:float(cond(v)@excess[bad]-1),0.,100.,xtol=1e-13)
independent=np.empty_like(mu);independent[bad]=.1*cond(nu);independent[~bad]=.9*mu[~bad]/mu[~bad].sum()
assert max(abs(independent-z['tail_constrained']))<1e-8
assert np.max(independent@z['features']-z['old_bounds'])<0
# Conditional-relative-entropy decomposition independently checks global optimality:
# good conditional remains mu; bad conditional is the exponential tilt with mean excess=1.
bad_mass=float(mu[bad].sum());lambda_bad=float(np.log((.9/.1)*np.exp(logsumexp(np.log(mu[bad])-nu*excess[bad]))/mu[~bad].sum()))
assert lambda_bad>0 and nu>0
result=dict(status='verified',scalar_root=nu,bad_indicator_multiplier=lambda_bad,
    max_probability_difference=float(abs(independent-z['tail_constrained']).max()),
    bad_mass=float(independent@bad),mean_excess=float(independent@excess),
    old_constraints_strictly_satisfied=True,
    caveat='Main output remains first dual solution; scalar root verifies, does not select a better scientific candidate. Main q90 crosses a discrete jump due to approximately 1e-10 mass tolerance.',
    seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
(p/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
