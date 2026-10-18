"""FP64 clean-p0 role matching and signed two-endpoint logit gradients.

SciPy exact linear assignment (floating-point costs), no soft OT, role sampling,
node-ID feature, random tie breaker, posterior-statistic loss or neural gradient.
"""
import copy
import hashlib
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from stepwise_role.core import Layout, signature, transition_kl

SETTINGS = dict(version='dynamic_anonymous_joint_role_v1', delta_max=.1,
    max_trials=8, descent_atol=1e-12, descent_rtol=1e-9,
    gradient_floor=1e-14, finite_difference_atol=2e-8,
    finite_difference_rtol=2e-5, renumber_atol=2e-10,
    renumber_rtol=1e-9, posterior_atol=5e-14,
    assignment_atol=1e-12, assignment_rtol=1e-10)


def freeze(profile, layout, alpha, reference_sha256):
    """Canonical anonymous multiset only; no original node assignment survives."""
    from hsgd.kernel import MarginalKernel
    alpha = MarginalKernel(layout.prior, alpha).alpha
    if (profile.get('format') != 'anonymous-joint-relation-degree-v1' or
        profile.get('family_pairs') != layout.pairs.tolist() or
        set(profile.get('groups', {})) != set(layout.endpoints)):
        raise ValueError('target schema/type inventory differs')
    if len(reference_sha256) != 64 or any(c not in '0123456789abcdef' for c in reference_sha256):
        raise ValueError('explicit training-reference SHA256 required')
    groups, totals = {}, {}
    for c in sorted(profile['groups'], key=int):
        g = profile['groups'][c]; rows = np.asarray(g['rows']); counts = np.asarray(g['counts'])
        n = layout.sizes[int(c)]
        if (rows.ndim != 2 or rows.shape[1] != len(layout.pairs) or counts.shape != (len(rows),)
            or not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(counts.dtype, np.integer)
            or (rows < 0).any() or (counts <= 0).any() or counts.sum() != n or g['nodes'] != n
            or len(np.unique(rows, axis=0)) != len(rows)):
            raise ValueError('invalid integer joint-role multiset')
        cap = np.array([max(0, layout.sizes.get(int(b if int(c) == a else a), 0)-(a == b))
                        if int(c) in (a,b) else 0 for a,b in layout.pairs])
        if ((rows > cap).any() or (rows[:,layout.prior == 0] != 0).any() or
            (rows[:,layout.prior == 1] != cap[layout.prior == 1]).any()):
            raise ValueError('role outside capacity/exact support')
        order = sorted(range(len(rows)), key=lambda i: tuple(rows[i]))
        rows, counts = rows[order], counts[order]
        scales = 1+np.sqrt(np.sum(rows.astype(float)**2*counts[:,None], axis=0)/n)
        dims = np.flatnonzero((cap > 0) & (layout.prior > 0))
        groups[c] = dict(nodes=n, rows=rows.tolist(), counts=counts.tolist(),
                         scale=scales.tolist(), effective_dimensions=dims.tolist())
        totals[int(c)] = (rows*counts[:,None]).sum(0)
    for r,(a,b) in enumerate(layout.pairs):
        if a == b:
            if totals[int(a)][r] % 2: raise ValueError('odd homogeneous degree total')
        elif totals[int(a)][r] != totals[int(b)][r]:
            raise ValueError('cross-type endpoint degree totals disagree')
    payload = dict(format='dynamic-anonymous-joint-role-v1', groups=groups,
        family_pairs=layout.pairs.tolist(), prior=layout.prior.tolist(), alpha=alpha.tolist(),
        reference_sha256=reference_sha256, settings=SETTINGS.copy(),
        coordinate='clean p0; fixed matching during line search; no solver gradient',
        normalization='mean over incident active dimensions, nodes, then ALL types; empty dimension type contributes zero',
        support='complete canonical type-pair slots; prior=0/1 exact; node-specific masks unsupported')
    return dict(payload=payload, sha256=signature(payload))


def _tie_audit(cost, assignment, roles, check):
    """Dual-feasible difference potentials + equality SCC, NOT strict uniqueness.

    Columns are permuted to the selected assignment. With W_ij=C_i,sigma(j)
    - C_i,sigma(i), shortest-path potentials certify reduced costs >= -tol.
    Alternating equality cycles changing role VALUE imply alternative gradients.
    Numerical equality is explicitly distinguished from exact algebraic equality.
    """
    n = len(assignment); tol = SETTINGS['assignment_atol']+SETTINGS['assignment_rtol']*max(1., float(abs(cost).max(initial=0)))
    w = cost[:,assignment]-cost[np.arange(n),assignment,None]
    d = np.zeros(n)
    for iterations in range(n):
        check(); nd = np.minimum(d, np.min(d[:,None]+w, axis=0))
        improvement = float(np.max(d-nd, initial=0)); d = nd
        if improvement <= tol/16: break
    reduced = w+d[:,None]-d[None,:]
    residual = float(max(0., -reduced.min(initial=0)))
    gap = float(n*residual)
    graph = abs(reduced) <= tol
    _, labels = connected_components(csr_matrix(graph), directed=True, connection='strong')
    selected = roles[assignment]
    duplicate, distinct = [], []
    for lab in np.unique(labels):
        ix = np.flatnonzero(labels == lab)
        if len(ix) > 1:
            (distinct if len(np.unique(selected[ix],axis=0)) > 1 else duplicate).append(ix)
    witnesses=[]
    # Cheap exact two-cycle witness (larger alternating cycles stay unresolved).
    for ix in distinct:
        check()
        for i in ix:
            js=ix[np.any(selected[ix] != selected[i], axis=1)]
            candidates=js[graph[i,js] & graph[js,i]]
            if len(candidates):
                j=int(candidates[0]); delta=float(w[i,j]+w[j,i])
                witnesses.append(dict(rows=[int(i),j], objective_difference=delta,
                    exact_float_zero=(delta == 0.), roles=[selected[i].tolist(),selected[j].tolist()]))
                break
    if residual > tol:
        status='dual_audit_unresolved'
    elif distinct:
        status=('nonidentical_role_tie_witness' if any(x['exact_float_zero'] for x in witnesses)
                else 'numerically_indistinguishable_role_tie_witness' if witnesses else 'possible_nonidentical_tie_unresolved')
    elif duplicate:
        status='duplicate_roles_only_at_declared_resolution'
    else:
        status='separated_at_declared_resolution'
    # Sum_i C_i,sigma(i)+sum_i d_i-sum_j d_j, relaxed by n*residual,
    # gives an applicable dual lower bound even when reduced costs slightly <0.
    return dict(status=status, equality_tolerance=tol, dual_max_violation=residual,
        assignment_objective_lower_bound=float(cost[np.arange(n),assignment].sum()-gap),
        primal_dual_gap_bound=gap, potential_iterations=iterations+1,
        duplicate_only_components=len(duplicate), distinct_role_components=len(distinct),
        witnesses=witnesses, strict_uniqueness_claim=False,
        arithmetic_scope='FP64 cost/dual residual audit; not a rigorous interval certificate or exact-real uniqueness proof',
        tie_rule='SciPy current row order / canonical lexicographic expanded roles; no persistent node-role binding; tied gradients need not be equivariant')


def exact_match(mu, roles, scale, dims, *, method='auto', check=lambda:None):
    n = len(mu); d = len(dims)
    if roles.shape != mu.shape or not n or not np.isfinite(mu).all(): raise ValueError('invalid matching input')
    tick=time.perf_counter()
    if not d:
        assignment=np.arange(n); cost=np.zeros((n,n)); algorithm='zero-dimensional identity storage assignment'
    else:
        # Sum squared differences explicitly; avoid cancellation in ||x||²+||y||²-2xy.
        cost=np.zeros((n,n))
        for r in dims:
            check(); cost += ((mu[:,r,None]-roles[None,:,r])/scale[r])**2/d
        if d == 1 and method != 'generic':
            r=dims[0]; ii=np.argsort(mu[:,r],kind='stable'); jj=np.argsort(roles[:,r],kind='stable')
            assignment=np.empty(n,int); assignment[ii]=jj; algorithm='exact 1D rearrangement sorting'
        else:
            check(); ii,jj=linear_sum_assignment(cost); assignment=np.empty(n,int); assignment[ii]=jj
            algorithm='scipy.optimize.linear_sum_assignment (modified Jonker-Volgenant)'
    check(); audit=_tie_audit(cost,assignment,roles,check)
    upper=float(cost[np.arange(n),assignment].sum())
    if audit['dual_max_violation'] > audit['equality_tolerance']:
        raise FloatingPointError('assignment dual audit failed at frozen tolerance')
    return assignment, dict(algorithm=algorithm, nodes=n, distinct_roles=len(np.unique(roles,axis=0)),
        role_copies=n, effective_dimensions=d, cost_sum=upper, normalized_type_loss=upper/n,
        seconds=time.perf_counter()-tick, **audit)


class MatchingGuide:
    def __init__(self, layout, artifact):
        self.layout=layout
        if signature(artifact['payload']) != artifact['sha256']: raise ValueError('target artifact hash mismatch')
        p=artifact['payload']
        profile=dict(format='anonymous-joint-relation-degree-v1',family_pairs=p['family_pairs'],
            groups={c:{k:g[k] for k in ('nodes','rows','counts')} for c,g in p['groups'].items()})
        if freeze(profile,layout,p['alpha'],p['reference_sha256']) != artifact: raise ValueError('artifact binding differs')
        self._artifact=copy.deepcopy(artifact); self.alpha=np.array(p['alpha'])
        self._groups={}
        for c,g in p['groups'].items():
            self._groups[c]=dict(nodes=np.flatnonzero(layout.types==int(c)),
                roles=np.repeat(np.array(g['rows'],float),g['counts'],axis=0),
                scale=np.array(g['scale']),dims=np.array(g['effective_dimensions'],int))
        self._initial=self._derived()

    @property
    def frozen(self): return copy.deepcopy(self._artifact)

    def _derived(self):
        def digest(a):return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
        return signature(dict(alpha=digest(self.alpha), groups={c:{k:digest(v) for k,v in g.items()} for c,g in self._groups.items()},
            layout={k:digest(getattr(self.layout,k)) for k in ('types','slots','families','pairs','prior','fixed0','fixed1')}))

    def binding(self):
        import scipy
        if signature(self._artifact['payload'])!=self._artifact['sha256'] or self._derived()!=self._initial:
            raise ValueError('target/layout/schedule mutation detected')
        return dict(target_sha256=self._artifact['sha256'], derived_sha256=self._initial,
                    scipy_runtime_version=scipy.__version__,numpy_runtime_version=np.__version__,
                    implementation_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())

    def expected_degrees(self,p):
        p=self.layout.probability(p); mu=np.zeros((len(self.layout.types),len(self.layout.pairs)))
        for end in self.layout.slots: np.add.at(mu,(end,self.layout.families),p)
        return mu

    def match(self,p,check=lambda:None):
        mu=self.expected_degrees(p); assignments={}; report={}
        for c,g in self._groups.items():
            check(); assignments[c],report[c]=exact_match(mu[g['nodes']],g['roles'],g['scale'],g['dims'],check=check)
        return assignments,report

    def loss_gradient(self,p,assignments,gradient=True):
        mu=self.expected_degrees(p); derivative=np.zeros_like(mu); loss=0.; detail={}
        C=len(self._groups)
        for c,g in self._groups.items():
            nodes,dims=g['nodes'],g['dims']; assigned=g['roles'][assignments[c]]
            residual=(mu[nodes]-assigned)/g['scale']; weighted=np.zeros_like(residual)
            if len(dims):
                weighted[:,dims]=residual[:,dims]**2/(len(nodes)*len(dims))
                if gradient:
                    derivative[np.ix_(nodes,dims)]=2*residual[:,dims]/g['scale'][dims]/(C*len(nodes)*len(dims))
            value=float(weighted.sum()); loss+=value/C
            detail[c]=dict(loss=value, standardized_residual_rms_by_relation=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
                           expected_degree_mean=mu[nodes].mean(0).tolist(), effective_dimensions=dims.tolist())
        gp=None
        if gradient:
            gp=derivative[self.layout.slots[0],self.layout.families]+derivative[self.layout.slots[1],self.layout.families]
        if not np.isfinite(loss) or gradient and not np.isfinite(gp).all(): raise FloatingPointError('nonfinite clean-role loss/gradient')
        return loss,gp,detail

    def apply(self,p0,state,kernel,t,check_time=None):
        start=time.perf_counter()
        check=check_time or (lambda:None); self.binding(); check()
        if not np.array_equal(kernel.alpha,self.alpha) or not np.array_equal(kernel.prior,self.layout.prior): raise ValueError('posterior binding differs')
        if not 1<=t<len(self.alpha): raise ValueError('reverse time outside schedule')
        p=self.layout.probability(p0); f=self.layout.families
        tick=time.perf_counter();q0=kernel.reverse(p,state,f,t-1,t);posterior_seconds=time.perf_counter()-tick
        out,qout=p.copy(),q0.copy()
        free=(p>0)&(p<1)&~self.layout.fixed0&~self.layout.fixed1
        record=dict(status='skipped',time=int(t),next_time=int(t-1),coordinate='CLEAN p0 logits, not q',
            proposed_delta=float(.1*self.alpha[t-1]),accepted_delta=0.,trials=0,backtracks=0,
            effective_movable_slots=int(free.sum()),structural_fixed_slots=int((self.layout.fixed0|self.layout.fixed1).sum()),
            numerical_saturated_slots=int((~free & ~(self.layout.fixed0|self.layout.fixed1)).sum()),
            matching_seconds=0.,gradient_seconds=0.,backtracking_seconds=0.,new_sampling_law=True)
        try:
            tick=time.perf_counter(); assignments,audit=self.match(p,check)
            record.update(matching_seconds=time.perf_counter()-tick,matching=audit)
            tick=time.perf_counter(); loss,gp,detail=self.loss_gradient(p,assignments)
            g=gp*p*(1-p); g[~free]=0; norm=float(abs(g).max(initial=0))
            if not np.isfinite(loss) or not np.isfinite(g).all():raise FloatingPointError('nonfinite clean-role logit gradient')
            threshold=SETTINGS['descent_atol']+SETTINGS['descent_rtol']*abs(loss)
            pred=record['proposed_delta']*float(g@g)/norm if norm else 0.
            if not np.isfinite(pred):raise FloatingPointError('nonfinite predicted descent')
            record.update(loss_before=loss,loss_after_fixed=loss,loss_after_reoptimized=loss,
                type_before=detail,type_after_fixed=detail,gradient_seconds=time.perf_counter()-tick,
                raw_logit_gradient_max=norm,raw_logit_gradient_l2=float(np.linalg.norm(g)),
                descent_threshold=threshold,first_order_descent_prediction=pred)
            if not free.any():record['reason']='no_movable_probabilities'
            elif norm<=SETTINGS['gradient_floor']:record['reason']='zero_or_roundoff_gradient'
            elif pred<=threshold:record['reason']='predicted_descent_below_resolution'
            else:
                z=np.log(p[free])-np.log1p(-p[free]); direction=g[free]/norm; tick=time.perf_counter()
                for j in range(8):
                    check(); step=record['proposed_delta']/(2**j); candidate=p.copy()
                    candidate[free]=np.exp(-np.logaddexp(0.,-(z-step*direction)))
                    after,_,after_detail=self.loss_gradient(candidate,assignments,False)
                    record.update(trials=j+1,backtracks=j,trial_loss=after)
                    if after<loss-threshold:
                        out=candidate;post_tick=time.perf_counter();qout=kernel.reverse(out,state,f,t-1,t)
                        posterior_seconds+=time.perf_counter()-post_tick
                        record.update(status='accepted',reason='resolved_fixed_matching_decrease',
                            loss_after_fixed=after,type_after_fixed=after_detail,accepted_delta=step)
                        break
                else:record['reason']='no_verified_descent'
                record['backtracking_seconds']=time.perf_counter()-tick
            check(); tick=time.perf_counter(); rematch,re_audit=self.match(out,check)
            re_loss,_,re_detail=self.loss_gradient(out,rematch,False)
            allowance=SETTINGS['assignment_atol']+SETTINGS['assignment_rtol']*max(1.,abs(record['loss_after_fixed']))
            if re_loss>record['loss_after_fixed']+allowance: raise FloatingPointError('reoptimized loss exceeds feasible fixed assignment')
            record.update(loss_after_reoptimized=re_loss,type_after_reoptimized=re_detail,
                reoptimized_matching=re_audit,reoptimization_seconds=time.perf_counter()-tick,
                reoptimization_excess=re_loss-record['loss_after_fixed'],reoptimization_allowance=allowance)
        except FloatingPointError as exc:
            out,qout=p.copy(),q0.copy();record.update(status='numerical_failure',reason=str(exc))
        record.update(clean_probability_l1=float(abs(out-p).sum()),clean_probability_max=float(abs(out-p).max(initial=0)),
            posterior_seconds=posterior_seconds,
            numerical_saturated_slots_after=int((((out==0)|(out==1)) & ~(self.layout.fixed0|self.layout.fixed1)).sum()),
            transition_probability_l1=float(abs(qout-q0).sum()),transition_probability_max=float(abs(qout-q0).max(initial=0)),
            transition_kl=transition_kl(qout,q0),wall_seconds=time.perf_counter()-start,
            statistics_meaning='expected clean degrees matched to hard anonymous roles; NOT sampled degree-distribution matching')
        return out,qout,record
