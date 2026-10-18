"""Built-in four-slot ENGINEERING fixture for the public CLI, not six states.

No model/weights, graph sampling, scientific input, or alternative mathematics.
The wait fixture intentionally emulates a child stuck outside cooperative
checks so the unchanged parent supervisor's timeout/STOP cleanup is exercised.
"""
import time
import numpy as np
from .core import Layout,MatchingGuide,freeze
from hsgd.kernel import MarginalKernel
from role_degree.role_statistics import joint_profile
from multiscale_role.runtime_guard import atomic_json,checked_sha


def run(args,out,check):
    check()
    atomic_json(out/'smoke_started.json',dict(scope='ENGINEERING_ONLY',mode=args.smoke_mode),check)
    if args.smoke_mode=='wait':
        # No scientific computation here. Parent must terminate/reap this
        # noncooperative fixture within its ORIGINAL inclusive budget.
        while True:time.sleep(.025)
    layout=Layout.canonical([0,0,1,2],[(0,1),(0,2)],[.5,.5])
    clean=np.array([[0,0],[2,3]],dtype=np.int64)
    profile=joint_profile(layout.types,clean,np.array([0,1]),layout.pairs)
    kernel=MarginalKernel(layout.prior,[1.,.8,.4,0.])
    guide=MatchingGuide(layout,freeze(profile,layout,kernel.alpha,'0'*64))
    p=np.array([.17,.63,.46,.81]);state=np.array([1,0,0,1],dtype=np.uint8)
    if args.smoke_mode=='invalid':p=np.array([1.1,.63,.46,.81])
    pp,q,record=guide.apply(p,state,kernel,1,check)
    checks=dict(resolved_descent=record['status']=='accepted',
        finite_probabilities=bool(np.isfinite(pp).all() and np.isfinite(q).all()),
        exact_t1_correspondence=bool(np.array_equal(pp,q)),
        reoptimized_not_above_fixed=bool(record['loss_after_reoptimized']<=record['loss_after_fixed']+record['reoptimization_allowance']))
    if not all(checks.values()):raise ValueError('toy engineering smoke check failed')
    atomic_json(out/'engineering_result.json',dict(scope='ENGINEERING_ONLY_NOT_SIX_STATES',
        checks=checks,p0_before=p.tolist(),p0_after=pp.tolist(),q=q.tolist(),guidance=record),check)
    return dict(status='complete',scope='engineering_cli_smoke',checks=checks,
        artifact_sha256=checked_sha(out/'engineering_result.json',check),
        model_calls=0,training_updates=0,scientific_graphs=0,GPU_use=0)
