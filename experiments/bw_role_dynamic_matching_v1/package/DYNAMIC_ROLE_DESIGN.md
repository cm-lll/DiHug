# Dynamic anonymous joint-role matching v1

Base: **7049e64b108f2ef434c40815457c74037c4455eb**. This directory is independent.
All 121 baseline package files are preserved byte-for-byte; only new files are
added. Historical entry points remain historical, not the new method's launcher.
No training, full scientific chain, model-weight import, GPU or BW execution is
authorized by these entry points. Small neural fixtures are engineering tests.

## Anonymous target and exact hard assignment

For type c, store distinct complete R-dimensional integer degree vectors v_cj
and positive integer frequencies k_cj summing to n_c. Store no node assignment,
node ID or edge identity. The artifact binds the reference hash and schema.
Canonical lexicographic row ordering is serialization, not a fixed node role.

S_cr = 1 + sqrt(sum_j k_cj v_cjr^2 / n_c), frozen from training reference.
I_c contains incident relations with positive capacity and positive prior.
Exact-one prior dimensions remain included (their residual is identically zero);
zero-prior dimensions remain in full vectors but not the normalization. All C
node types have equal weight, including zero-dimensional types (zero loss).

mu_ir = sum_{e incident to i, family(e)=r} p0_e.

L(p0,sigma) = (1/C) sum_c (1/(n_c |I_c|))
                 sum_{i:type(i)=c} sum_{r in I_c}
                    ((mu_ir - v_c,sigma(i),r)/S_cr)^2.

Minimize jointly within each type, subject to each expanded role copy being used
exactly once. Never separately sort multiple relation dimensions. SciPy
`linear_sum_assignment` is the trusted exact combinatorial algorithm on the
computed FP64 costs (modified Jonker–Volgenant); one effective dimension uses
the rearrangement inequality and stable ascending sort. Counts are expanded,
not soft-averaged. Cost workspace O(max_c n_c^2), solver worst-case cubic;
full probabilities/expectations remain O(S+NR). This is not a learned N×N head
and not a claim of end-to-end linear space.

The implementation audits a dual lower bound of its FP64 cost matrix through
difference-constraint potentials and equality-graph SCCs. This is a residual
audit, not rigorous interval arithmetic or proof of strict real uniqueness.
Cost tolerance = 1e-12 + 1e-10 * max(1, max absolute cost). Any dual residual
above this limit is a numerical failure, not a silent fallback counted as success.
The reported lower bound subtracts n times the maximum dual violation; original
and normalized objective values and audit limits remain visible.

## Derivative and controller

One new optimal assignment is computed for each call/timestep; stop gradient
through the solver and retain this assignment during every backtracking trial.
For e=(u,v), r=family(e), define

b_ir = 2 (mu_ir - assigned_role_ir) / (C n_type(i) |I_type(i)| S_type(i),r^2).

dL/dp0_e = b_ur + b_vr, including both endpoints for same-type edges.
dL/dz_e = (dL/dp0_e) p0_e (1-p0_e), z=logit(p0).

These are signed residual contributions, not two positive rewards. The loss is
on CLEAN p0, NOT q and NOT the next noisy target used by the old CF objective.

Preserve max step delta_t=.1*alpha[t-1], infinity-norm gradient normalization,
8 candidates delta_t/2^j (j=0..7), descent threshold 1e-12+1e-9*abs(L),
gradient floor1e-14 and predicted-descent reliability gate. No new tuning.
Only a resolved fixed-assignment decrease accepts. Recompute optimum afterward
for audit, never for candidate selection; it must be <= fixed loss within the
declared assignment allowance. All skip/failure reasons are retained.
No epsilon, clipping, flips, repair, inertia, random ties or evaluation feedback.
Structural prior0/1 and numerical exact0/1 are fixed and counted separately.

## Sampling interface

`dynamic_role.sampling.reverse_step` reuses the original stepwise adapter:
freeze full Xt → original20 query blocks → collect p0 → release neural graphs →
match and adjust clean logits → ORIGINAL Bayes posterior q → original saved
uniforms → atomic whole-state commit. Recompute at next time. No hidden-state
cache across blocks; no new sampling RNG consumption. Disabled A/B dispatch
unchanged original runtime. A/C share native no-CN weights; B/D share native CN
weights. A is the joint-framework no-CN control, not old v9. No weight file is
loaded by the offline diagnostic. Native runtime and checkpoint bindings include
target, layout, model, code, schedule, backend, query seed, runtime and RNG.

## Ties: a real limitation, not silently repaired

Identical role copies can be interchanged without changing residuals. Distinct
roles at equal optimal cost can yield different gradients. Stable sort / SciPy
uses current row storage order for a representative solution; there is no
permanent reference-ID mapping. The equality-SCC audit distinguishes duplicate-
only components, exact-zero FP64 two-cycle witnesses, numerically unresolved
two-cycles and unresolved larger cycles. A missing witness is NOT a uniqueness
proof. Unique/separated small cases are checked under type-preserving relabeling.
An explicitly symmetric distinct-role counterexample is retained: objective is
invariant, selected gradient need not be equivariant. No randomized tie-breaker
or averaging of hard roles is introduced. This limitation must be assessed before
scientific deployment, especially high-noise symmetric predictions.

## Meaning and risks

This matches expected CLEAN relation degrees to roles; it is NOT a distance
between sampled degree distributions and not exact conditional generation.
Posterior means can legitimately shrink at high noise; hard one-to-one targets
may force premature differentiation or arbitrary symmetry breaking. Even if
proxy decreases, degree variance, topology, connectivity and closure may worsen.
Independent Genre64 and exact degree-law W1 are evaluation only. No CF/tail
term remains in the new objective; old modules are only preserved/evaluation.

## Frozen tolerances and scientific decision

Finite difference: atol2e-8/rtol2e-5. Unique-case relabeling:
atol2e-10/rtol1e-9. Saved original Bayes q: atol5e-14.
Metric baseline reproduction/decisions: CF atol2e-10, W1 atol1e-8,
rtol1e-9, existing full-PMF operation-count allowances explicitly retained.
Old64 frequency bank, original normalizer1, unscaled W1 of expected empirical
integer-degree PMFs are unchanged. Not E[W1 of a sampled graph].

B1 is development first; six-state entry requires a successful B1 receipt with
identical implementation, target and portable-input hashes. It then freezes
ALL A/B×500/20/1, no tuning. A valid full panel requires all6 complete;
all skipped means no effect. Any resolved Genre64 or Genre-W1 worsening means
tradeoff/deterioration, not overall repair. Only no worsening in any state plus
at least one dual improvement supports a limited frozen-panel gain. No result
proves full-chain stability, generated quality or unseen-reference generalization.

Full exact PMF evaluation remains O(nodes*capacity^2), potentially dominant.
Do not silently drop channels, approximate, truncate or enlarge900 seconds.
Timeout/STOP leaves partial evidence explicitly incomplete. There is no old
absolute deadline. Parent cleans/reaps only its child group for INT/TERM/HUP,
STOP/timeout; SIGKILL and machine loss cannot execute Python cleanup.
