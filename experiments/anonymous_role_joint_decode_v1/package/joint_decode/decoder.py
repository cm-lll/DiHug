"""FIXED full role matching MAP only; not anonymous global MAP or a sampler.

Cross-type blocks: TU bipartite degree LP (min-cost-flow formulation).
Same-type blocks: binary simple-graph f-factor MILP. All endpoints contribute.
"""
import math
import time
import warnings
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from .contracts import CONFIG,validate_assignment

OPTIONS=CONFIG['decoder']


def decode(schema,target,assigned,law,*,seconds=10.,check=lambda:None,event=lambda value:None):
    start=time.monotonic();deadline=start+seconds
    assigned=np.asarray(assigned);validate_assignment(schema,target,assigned)
    if len(law.logits)!=schema.s:raise ValueError('law/schema mismatch')
    bits=(law.support==1).astype(np.uint8);residual=assigned-schema.degree(bits)
    report=dict(scope='MAP under ONE FIXED COMPLETE role matching',status='incomplete',blocks=[],
        solver_options=OPTIONS,optimal=False,tie_rule='solver representative; no tied-output equivariance claim',
        hard_zero_slots=int((law.support==0).sum()),hard_one_slots=int((law.support==1).sum()),
        finite_logit_free_slots=int(law.free.sum()))
    if (residual<0).any():report.update(status='infeasible',reason='mandatory support exceeds assigned degree');return None,report
    total_lower=0.;all_optimal=True
    def guard():
        check()
        if time.monotonic()>=deadline:raise TimeoutError('fixed matching decoder time limit')
    try:
        for r,(a,b) in enumerate(schema.pairs):
            guard();tick=time.monotonic();family=np.flatnonzero(schema.families==r)
            u,v=schema.slots[:,family]
            eligible=law.free[family]&(residual[u,r]>0)&(residual[v,r]>0)
            ids=family[eligible];u,v=schema.slots[:,ids];need=residual[:,r]
            available=np.bincount(np.r_[u,v],minlength=schema.n)
            row=dict(family=r,pair=[int(a),int(b)],original_slots=len(family),free_variables=len(ids),
                residual_edges=int(need.sum()//2),algorithm='bipartite TU LP' if a!=b else 'binary same-type f-factor MILP')
            if (need>available).any() or need.sum()%2:
                row.update(status='infeasible',reason='insufficient allowed neighbors or parity');report['blocks'].append(row)
                report.update(status='infeasible',reason='fixed role matching conflicts with support');return None,report
            if not len(ids):
                if need.any():raise AssertionError('zero-variable nonzero residual escaped check')
                row.update(status='optimal',objective=0.,lower_bound=0.,gap=0.,seconds=time.monotonic()-tick)
                report['blocks'].append(row);event(row);continue
            A=csc_matrix((np.ones(2*len(ids)),(np.r_[u,v],np.tile(np.arange(len(ids)),2))),shape=(schema.n,len(ids)))
            c=-law.logits[ids];remaining=max(.001,deadline-time.monotonic())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if a!=b:
                    options=dict(time_limit=remaining,presolve=True,threads=OPTIONS['threads'],
                        primal_feasibility_tolerance=OPTIONS['lp_primal_feasibility_tolerance'],
                        dual_feasibility_tolerance=OPTIONS['lp_dual_feasibility_tolerance'])
                    sol=linprog(c,A_eq=A,b_eq=need,bounds=(0.,1.),method='highs-ds',options=options)
                else:
                    options=dict(time_limit=remaining,presolve=True,threads=OPTIONS['threads'],
                        mip_rel_gap=OPTIONS['mip_rel_gap'],node_limit=OPTIONS['mip_node_limit'])
                    sol=milp(c,integrality=np.ones(len(ids)),bounds=Bounds(0.,1.),
                        constraints=LinearConstraint(A,need,need),options=options)
            guard();row.update(solver_status=int(sol.status),solver_message=str(sol.message),
                warnings=[str(w.message) for w in caught],seconds=time.monotonic()-tick)
            if sol.x is None:
                row['status']='infeasible' if sol.status==2 else 'incomplete';report['blocks'].append(row)
                report.update(status=row['status'],reason='solver supplied no graph');return None,report
            if not np.isfinite(sol.x).all() or (sol.x < -OPTIONS['external_integrality_atol']).any() or (sol.x > 1+OPTIONS['external_integrality_atol']).any():
                row.update(status='numerical_failure',reason='nonfinite/out-of-bounds solver vector');report['blocks'].append(row)
                report.update(status='numerical_failure',reason='invalid solver vector');return None,report
            error=float(abs(sol.x-np.rint(sol.x)).max(initial=0));rounded=np.rint(sol.x).astype(np.int8)
            if error>OPTIONS['external_integrality_atol'] or not np.isin(rounded,[0,1]).all() or not np.array_equal(A@rounded,need):
                row.update(status='numerical_failure',integrality_error=error);report['blocks'].append(row)
                report.update(status='numerical_failure',reason='solver vector failed independent binary/degree audit');return None,report
            objective=float(c@rounded)
            if a!=b:
                y=np.asarray(sol.eqlin.marginals) if sol.eqlin.marginals is not None else None
                # Independent applicable Lagrangian lower bound for ANY y:
                # b.y + sum min(0,c-A'y), over 0<=x<=1. No dual feasibility
                # assumption or solver.success field substitutes for this audit.
                if y is None or y.shape!=(schema.n,) or not np.isfinite(y).all():lower=None
                else:
                    reduced=c.astype(np.longdouble)-y[u].astype(np.longdouble)-y[v].astype(np.longdouble)
                    lower=float(np.sum(need.astype(np.longdouble)*y.astype(np.longdouble),dtype=np.longdouble)+np.minimum(0,reduced).sum(dtype=np.longdouble))
                certificate='TU vertex integrality + independently recomputed Lagrangian lower bound (FP arithmetic)'
            else:
                bound=getattr(sol,'mip_dual_bound',None);lower=float(bound) if bound is not None and np.isfinite(bound) else None
                certificate='HiGHS branch-and-bound bound; independent exact binary degree feasibility; not interval arithmetic'
            gap=None if lower is None else objective-lower
            tolerance=OPTIONS['objective_atol']+OPTIONS['objective_rtol']*max(1.,abs(objective))
            if gap is not None and gap < -tolerance:
                row.update(status='numerical_failure',objective=objective,lower_bound=lower,gap=gap)
                report['blocks'].append(row);report.update(status='numerical_failure',reason='invalid objective bound');return None,report
            optimal=sol.status==0 and gap is not None and abs(gap)<=tolerance
            bits[ids]=rounded;all_optimal &= optimal
            total_lower=total_lower+lower if total_lower is not None and lower is not None else None
            row.update(status='optimal' if optimal else 'feasible_unproven',objective=objective,lower_bound=lower,
                gap=gap,objective_tolerance=tolerance,integrality_error=error,
                independent_degree_error=0,certificate=certificate,
                node_count=int(getattr(sol,'mip_node_count',0) or 0))
            report['blocks'].append(row);event(row)
        guard()
        degree=schema.degree(bits)
        if not np.array_equal(degree,assigned) or not law.allowed(bits):raise AssertionError('whole-graph feasibility audit failed')
        validate_assignment(schema,target,degree)
        score=float(law.log_prob(bits));bound=None if total_lower is None else law.log_base-total_lower
        report.update(status='optimal_fixed_matching' if all_optimal else 'feasible_unproven',optimal=bool(all_optimal),
            exact_joint_role_multiset=True,exact_assigned_degrees=True,exact_support=True,
            log_q=score,log_q_upper_bound=bound,log_q_gap=None if bound is None else bound-score,
            physical_edges=int(bits.sum()),seconds=time.monotonic()-start,
            no_anonymous_global_optimality_claim=True,no_sampling_claim=True)
        return bits,report
    except TimeoutError as exc:
        report.update(status='incomplete',reason=str(exc),seconds=time.monotonic()-start)
        return None,report
