"""Float64 conditional-product statistic and ORIGINAL-posterior logit gradient.

All arrays are full legal unordered slots. No network gradient, node assignment,
sampling RNG, epsilon, probability clipping, truncation or graph repair.
"""
import copy
import hashlib
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'role_degree'))
from role_guidance import Layout, freeze_target, signature

SETTINGS = dict(version='stepwise_role_v1', delta_max=.1, max_trials=8,
                descent_atol=1e-12, descent_rtol=1e-9, gradient_floor=1e-14,
                finite_difference_atol=2e-8, finite_difference_rtol=2e-5,
                renumber_atol=2e-10, renumber_rtol=1e-9,
                posterior_atol=5e-14, frequency_bound=.5)


def freeze(profile, layout, alpha):
    from hsgd.kernel import MarginalKernel
    alpha = MarginalKernel(layout.prior, alpha).alpha
    old = freeze_target(profile, layout)  # EXACT old K32, seed, scales and rule
    capacities = {}
    for c in layout.sizes:
        capacities[str(c)] = [int(layout.sizes.get(int(b if c == a else a), 0) - (a == b))
                              if c in (a, b) else 0 for a, b in layout.pairs]
    payload = dict(settings=SETTINGS.copy(), frequencies=old, alpha=alpha.tolist(),
                   prior=layout.prior.tolist(), capacities=capacities,
                   legal_support='complete type-pair slots; node-specific masks unsupported',
                   time='t descends T..1; s=t-1; clean=0',
                   coordinate='logit of collected supported p0; numerical 0/1 stay fixed',
                   condition='known training-reference anonymous joint distribution')
    return dict(payload=payload, sha256=signature(payload))


def transition_kl(p, q):
    """KL(Bernoulli(p)||Bernoulli(q)), natural log, exact limits, no epsilon."""
    p, q = np.asarray(p), np.asarray(q)
    if ((q == 0) & (p > 0) | (q == 1) & (p < 1)).any():
        raise FloatingPointError('guided transition left baseline support')
    value = np.zeros_like(p, dtype=np.float64)
    for a, b in ((p, q), (1-p, 1-q)):
        take = a > 0
        value[take] += a[take] * (np.log(a[take])-np.log(b[take]))
    # Keep roundoff visible, do not silently clamp a tiny negative KL.
    return dict(sum=float(value.sum()), max_slot=float(value.max(initial=0)),
                minimum_slot=float(value.min(initial=0)),
                summation_roundoff_bound=float(32*np.finfo(float).eps*len(p)))


class StepwiseGuide:
    def __init__(self, layout, frozen):
        self.layout = layout
        if signature(frozen['payload']) != frozen['sha256']:
            raise ValueError('guidance artifact hash mismatch')
        expected = freeze(frozen['payload']['frequencies']['payload']['profile'], layout,
                          frozen['payload']['alpha'])
        if expected != frozen:
            raise ValueError('target/frequency/config/schedule/prior binding mismatch')
        self.frozen = copy.deepcopy(frozen)
        self.settings = dict(SETTINGS)
        self.alpha = np.array(frozen['payload']['alpha'])
        source = frozen['payload']['frequencies']['payload']
        self.omega = {c: np.array(w) for c, w in source['omega'].items()}
        self.targets = {}
        for c, g in source['profile']['groups'].items():
            rows = np.asarray(g['rows'], dtype=np.int64)
            counts = np.asarray(g['counts'], dtype=np.float64) / g['nodes']
            cap = np.asarray(frozen['payload']['capacities'][c])
            w = self.omega[c]
            if np.abs(w).max(initial=0) > self.settings['frequency_bound']:
                raise ValueError('unguarded frequency range')
            h = np.expm1(1j*w)
            result = np.empty((len(self.alpha), len(w)), dtype=np.complex128)
            for s, a in enumerate(self.alpha):
                b = a + (1-a)*layout.prior
                z = (1-a)*layout.prior
                # Powers 0 included exactly. No log of zero; bound above
                # guarantees nonzero individual factors for all valid p.
                products = ((1+b*h)[None, :, :] ** rows[:, None, :] *
                            (1+z*h)[None, :, :] ** (cap-rows)[:, None, :]).prod(axis=2)
                result[s] = (counts[:, None]*products).sum(axis=0)
            self.targets[c] = result
        for v in [self.alpha, *self.omega.values(), *self.targets.values()]:
            v.flags.writeable = False

    def binding(self):
        # Recheck public mutable dictionaries so later mutation cannot silently
        # change a resumed law. Arrays themselves are readonly.
        if self.settings != SETTINGS or signature(self.frozen['payload']) != self.frozen['sha256']:
            raise ValueError('guidance configuration changed after freezing')
        def digest(v):
            return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest())
        derived = dict(alpha=digest(self.alpha), omega={c:digest(w) for c,w in self.omega.items()},
                       target={c:digest(v) for c,v in self.targets.items()},
                       layout={k:digest(getattr(self.layout,k)) for k in ('types','slots','families','pairs','prior')})
        return dict(frozen_sha256=self.frozen['sha256'], derived_sha256=signature(derived))

    def statistic(self, q, s, gradient=False):
        """E[empirical CF] under independent edge variables, NOT CF(E[degree]).

        Workspace O(S+N), K loop. A single unordered edge contributes to both
        endpoints. Factors have real part >=cos(.5)>0; explicitly check this
        BEFORE division. Never divide by a product (which may underflow).
        """
        q = self.layout.probability(q)
        if not 0 <= s < len(self.alpha): raise ValueError('next-state time outside schedule')
        gp = np.zeros_like(q) if gradient else None
        loss, underflow, small, minimum = 0., 0, 0, 0.
        means = {}
        K, C = 32, len(self.targets)
        for c, (ids, node) in self.layout.endpoints.items():
            n = self.layout.sizes[int(c)]
            f, values = self.layout.families[ids], q[ids]
            means[c] = []
            for k in range(K):
                h = np.expm1(1j*self.omega[c][k])[f]
                factor = 1+values*h
                if (factor.real < np.cos(.5)-1e-14).any() or not np.isfinite(factor).all():
                    raise FloatingPointError('factor lower bound violated; refusing division')
                logs = np.log1p(values*h)
                lr = np.bincount(node, weights=logs.real, minlength=n)
                li = np.bincount(node, weights=logs.imag, minlength=n)
                phi = np.exp(lr+1j*li)
                mean = phi.mean(); means[c].append(mean)
                residual = mean-self.targets[c][s,k]
                loss += float(abs(residual)**2)/(C*K)
                underflow += int((phi == 0).sum())
                small += int((np.abs(phi) < 1e-12).sum())
                minimum = min(minimum, float(lr.min(initial=0)))
                if gradient:
                    v = 2/(C*K*n)*np.real(np.conj(residual)*phi[node]*h/factor)
                    gp += np.bincount(ids, weights=v, minlength=len(q))
        if not np.isfinite(loss) or gp is not None and not np.isfinite(gp).all():
            raise FloatingPointError('nonfinite statistic/gradient')
        return loss, gp, dict(underflow_node_frequencies=underflow,
                              small_node_frequencies=small, min_log_modulus=minimum,
                              mean_cf=means)

    def posterior_gradient(self, p0, state, kernel, t):
        """Original reverse is affine in p0: q=b0+(b1-b0)p0.

        Evaluate its slope using the ORIGINAL implementation on feasible basis
        inputs. Structural fixed support stays fixed. Impossible conditioning
        is rejected by kernel.reverse, never assigned a made-up posterior.
        """
        if not np.array_equal(kernel.alpha, self.alpha) or not np.array_equal(kernel.prior, self.layout.prior):
            raise ValueError('posterior/schedule/prior mismatch')
        p = self.layout.probability(p0)
        f = self.layout.families
        q = kernel.reverse(p, state, f, t-1, t)
        lo = np.zeros_like(p); hi = np.ones_like(p)
        lo[self.layout.fixed1] = 1; hi[self.layout.fixed0] = 0
        b0 = kernel.reverse(lo, state, f, t-1, t)
        b1 = kernel.reverse(hi, state, f, t-1, t)
        error = float(np.max(np.abs(q-(b0+(b1-b0)*p)), initial=0))
        if error > self.settings['posterior_atol']:
            raise FloatingPointError('posterior affine check exceeded predeclared tolerance')
        loss, gp, detail = self.statistic(q, t-1, True)
        g = gp*(b1-b0)*p*(1-p)
        g[self.layout.fixed0 | self.layout.fixed1] = 0.
        return q, loss, g, detail, error

    def apply(self, p0, state, kernel, t, check_time=None):
        p = self.layout.probability(p0)
        if not 1 <= t < len(self.alpha): raise ValueError('reverse time outside schedule')
        started = time.perf_counter()
        q0 = kernel.reverse(p, state, self.layout.families, t-1, t)
        guided_p, guided_q = p.copy(), q0.copy()
        free = (p>0)&(p<1)&~self.layout.fixed0&~self.layout.fixed1
        record = dict(status='skipped', time=int(t), next_time=int(t-1),
            delta_max=self.settings['delta_max'], proposed_delta=float(self.settings['delta_max']*self.alpha[t-1]),
            accepted_delta=0., trials=0, backtracks=0, effective_movable_slots=int(free.sum()),
            structural_fixed_slots=int((self.layout.fixed0|self.layout.fixed1).sum()),
            numerical_saturated_slots=int(((p==0)|(p==1)).sum()-(self.layout.fixed0|self.layout.fixed1).sum()),
            configuration_sha256=self.frozen['sha256'], statistics_gradient_seconds=0.,
            backtracking_seconds=0., new_sampling_law=True)
        try:
            if check_time: check_time()
            tick = time.perf_counter()
            _, loss, g, detail, error = self.posterior_gradient(p, state, kernel, t)
            detail.pop('mean_cf')
            record.update(loss_before=loss, loss_after=loss, posterior_affine_error=error,
                          statistics_gradient_seconds=time.perf_counter()-tick, **detail)
            norm = float(np.max(np.abs(g), initial=0))
            threshold = self.settings['descent_atol']+self.settings['descent_rtol']*abs(loss)
            predicted = record['proposed_delta']*float(np.dot(g,g))/norm if norm else 0.
            record.update(raw_logit_gradient_max=norm, raw_logit_gradient_l1=float(np.abs(g).sum()),
                          raw_logit_gradient_l2=float(np.linalg.norm(g)), descent_threshold=threshold,
                          first_order_descent_prediction=predicted)
            # Conservative reliability gates; do not amplify underflowed or
            # unresolvable gradients into a full-size normalized step.
            if record['underflow_node_frequencies']: record['reason']='underflow_untrusted_gradient'
            elif not free.any(): record['reason']='no_movable_probabilities'
            elif norm <= self.settings['gradient_floor']: record['reason']='zero_or_roundoff_gradient'
            elif predicted <= threshold: record['reason']='predicted_descent_below_resolution'
            else:
                z = np.log(p[free])-np.log1p(-p[free])
                direction = g[free]/norm
                tick = time.perf_counter()
                for j in range(self.settings['max_trials']):
                    if check_time: check_time()
                    step = record['proposed_delta']/(2**j)
                    candidate = p.copy()
                    candidate[free] = np.exp(-np.logaddexp(0., -(z-step*direction)))
                    q = kernel.reverse(candidate, state, self.layout.families, t-1, t)
                    after, _, _ = self.statistic(q, t-1)
                    record.update(trials=j+1, backtracks=j, trial_loss=after)
                    if after < loss-threshold:
                        guided_p, guided_q = candidate, q
                        record.update(status='accepted', reason='resolved_proxy_decrease',
                                      loss_after=after, accepted_delta=step)
                        break
                else: record['reason']='no_verified_descent'
                record['backtracking_seconds']=time.perf_counter()-tick
        except FloatingPointError as exc:
            guided_p, guided_q = p.copy(), q0.copy()
            record.update(status='numerical_failure', reason=str(exc))
        record.update(clean_probability_l1=float(np.abs(guided_p-p).sum()),
                      clean_probability_max=float(np.abs(guided_p-p).max(initial=0)),
                      transition_probability_l1=float(np.abs(guided_q-q0).sum()),
                      transition_probability_max=float(np.abs(guided_q-q0).max(initial=0)),
                      transition_kl=transition_kl(guided_q, q0),
                      wall_seconds=time.perf_counter()-started)
        return guided_p, guided_q, record
