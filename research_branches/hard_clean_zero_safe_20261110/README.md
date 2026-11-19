# Zero-feedback-safe hard-risk combination

This isolated candidate repairs the **combination rule**, not the definition of
structural quality. The reference-relative relation count, anonymous joint
relation degree, present/absent typed triangle and component costs are unchanged.
The original A network, noise kernel, sampling kernel, AdamW and EMA are unchanged.
No CE is added to the structural update. The original pilot and failures remain
under `hard_clean_pilot_v2_20261110`.

## Why change it

For sampled parameter-gradient estimates `g_i`, simplex minimum-norm MGDA chooses
`min ||sum lambda_i g_i||²`, with nonnegative coefficients summing to one. If any
estimate is zero, selecting that vertex is optimal. This is mathematically correct
for that problem. It is an unsuitable stopping interpretation when a finite hard
sample batch has identical costs for one objective: zero estimated feedback need
not mean the true risk has zero derivative, nor that other objectives are solved.

## Frozen replacement

Let `b = mean_i(g_i)` over the original, fixed active objective set. Solve

```
min_v  1/2 ||v-b||²
subject to g_i.T v >= 0 for every i.
```

The descent sign is `-v`. The solution is the Euclidean projection of `b` onto
a closed convex cone. A zero estimate stays in the constraints as `0 >= 0`;
it cannot force the projection to zero by itself. Genuine opposition can still
produce zero. No forced nonzero step, gradient norm equalization, ridge, objective
deletion, resampling or threshold search is used.

With `H_ij=g_i.T g_j` and `a=ones(m)/m`, the dual is
`min_{lambda>=0} 1/2 lambda.T H lambda + lambda.T H a`.
Then `v=G.T(a+lambda)`. We enumerate all at most 32 active sets and verify dual
feasibility, original normalized primal slack and complementarity. Degenerate
Gram matrices use a rank-aware least-squares solve; a failed solve is an error,
not a valid zero update. `rcond=1e-12`, relative KKT residual `<=1e-9`; actual FP32
combination dot products are separately recorded with floating roundoff bounds.

For the **supplied estimates**, `g_i.T(-v)<=0`, `b.T v=||v||²`, and
`||v||<=||b||`. These are first-order algebraic properties, not guarantees for true
expected losses, finite updates, or AdamW's actual parameter displacement. The
nonlinear combination of stochastic estimates is not an unbiased gradient of a
single fixed scalar loss. Existing disparate objective scales still matter.

AdamW receives `v` as its gradient and advances its original moments and step;
its actual displacement includes preconditioning, historical momentum and weight
decay. EMA advances normally. This version does NOT project AdamW displacement,
perform probe-based accept/reject, or restart the abandoned update-protection
route. A group with zero sampled gradient remains **unidentified by this batch**,
not protected. Component cost continues to be evaluated and reported.

## Numerically stable LOO

The old difference `(K*c-sum(c))/(K-1)` could give tiny spurious feedback for a
constant, positive cost. First compute `d=c-c[0]`, then use
`(K*d-sum(d))/(K-1)`. Algebraically identical, including the anchor sample, this
gives exactly zero for constant costs in floating arithmetic. It does not change
the estimator's expectation. The absent-triangle exact second-moment estimator
from v2 is retained; no zero-cost group is inferred to have zero true gradient.

## Bounded checks

CPU checks include 204 synthetic cases, an independent primal SLSQP solution,
exact enumeration of a three-edge LOO expectation, and every saved successful
Gram matrix from v2 (including engineering gates; not independent training runs).

The real check restores four original failed boundaries. It reuses their next
historical input and saved auxiliary RNG, makes one temporary update, and checks
raw predictions before/after on the six old development inputs with 32 paired
hard draws. These are conditional clean-graph predictions, NOT reverse-chain
outputs or fresh held-out confirmation. The failed batch is now saved explicitly.
Only raw predictions are evaluated; EMA state is saved, not claimed protected.

GPU ceiling: one free 2080Ti, 540 seconds including process loading; outer timeout
terminates this process only. No long training, full generation or automatic
follow-up. File reload is verified bitwise; ordinary GPU repeated calculations
retain the historical non-bitwise limitation and original tolerances.

## Reproduce

From repository root, use the existing `sparse_block` Python environment:

```
python research_branches/hard_clean_zero_safe_20261110/test_cpu.py
python research_branches/hard_clean_zero_safe_20261110/check_real.py --prepare
CUDA_VISIBLE_DEVICES=<free_gpu> timeout --signal=TERM --kill-after=5s 540s \
  python research_branches/hard_clean_zero_safe_20261110/check_real.py
```

`--prepare` hashes the original inputs and executed sources before model calls.
The GPU script refuses an existing per-condition output directory; choose an
isolated new result root for reproduction. Do not overwrite the current evidence.
Results: `results/hard_clean_zero_safe_20261110/`.
