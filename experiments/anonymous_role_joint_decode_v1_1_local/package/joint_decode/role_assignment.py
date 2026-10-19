"""One exact type-wise JOINT role matching from complete clean p0; no alternating search."""
import itertools
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import validate_target,validate_assignment


def match(schema,target,p0,check=lambda:None):
    start=time.monotonic();validate_target(schema,target);p0=np.asarray(p0,dtype=np.float64)
    if p0.shape!=(schema.s,) or not np.isfinite(p0).all() or ((p0<0)|(p0>1)).any():raise ValueError('complete clean probabilities required')
    mu=np.zeros((schema.n,schema.r));d=np.zeros((schema.n,schema.r),np.int64);rows=[];total=0.
    for endpoint in schema.slots:np.add.at(mu,(endpoint,schema.families),p0)
    for c,n in schema.sizes.items():
        check();nodes=np.flatnonzero(schema.types==c);g=target['groups'][str(c)]
        roles=np.repeat(np.asarray(g['rows'],np.int64),np.asarray(g['counts']),axis=0)
        scale=1+np.sqrt(np.mean(roles.astype(float)**2,axis=0));valid=np.flatnonzero(np.any(roles>0,axis=0));k=len(valid)
        x=mu[nodes];ties='all roles identical' if len(g['rows'])==1 else 'not exhaustively certified; index-based solver representative'
        if k==0:order=np.arange(n);algorithm='all zero roles'
        elif k==1:
            r=valid[0];source=np.argsort(x[:,r],kind='stable');role=np.argsort(roles[:,r],kind='stable')
            order=np.empty(n,int);order[source]=role;algorithm='exact single-effective-relation sorting'
            ties='different-role tie present' if any(x[a,r]==x[b,r] and roles[order[a],r]!=roles[order[b],r] for a,b in zip(source[:-1],source[1:])) else 'no different-role sorting tie'
        else:
            # Only n*n matrix; never n*n*k broadcast. Whole graph networks are not retained.
            cost=np.zeros((n,n))
            for r in valid:
                check();cost+=((x[:,r,None]-roles[None,:,r])/scale[r])**2/k
            a,order=linear_sum_assignment(cost)
            if not np.array_equal(a,np.arange(n)):raise AssertionError('assignment rows')
            algorithm='scipy linear_sum_assignment exact joint square cost'
            if n<=7:
                optimum=float(cost[np.arange(n),order].sum());solutions=set()
                for perm in itertools.permutations(range(n)):
                    if abs(float(cost[np.arange(n),perm].sum())-optimum)<=1e-12:
                        solutions.add(tuple(map(tuple,roles[list(perm)])))
                ties='different-role tie present' if len(solutions)>1 else 'unique role-vector allocation within1e-12'
        check();d[nodes]=roles[order]
        value=float(np.mean(np.sum(((x[:,valid]-d[nodes][:,valid])/scale[valid])**2,axis=1)/k)) if k else 0.
        total+=value/len(schema.sizes)
        rows.append(dict(type=int(c),nodes=n,effective_relations=valid.tolist(),scales=scale.tolist(),
                         duplicate_role_copies=n-len(g['rows']),cost=value,algorithm=algorithm,
                         different_role_ties=ties,tie_rule='stable sorting/SciPy index representative; no strict tied-gradient or assignment equivariance claim'))
    validate_assignment(schema,target,d)
    return d,dict(status='complete',cost=total,types=rows,seconds=time.monotonic()-start,
        normalization='mean over effective relation dimensions, nodes within type, then node types; scales=1+reference type RMS',
        expected_joint_degrees=mu.tolist(),scope='one complete matching fixed for all relation decoders; not anonymous global MAP')
