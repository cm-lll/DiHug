# Conditional feedback for the unchanged hard component-count risk

Scope: CPU analysis of stored conditional clean-graph probabilities and hard
samples. No denoiser invocation, new real samples, optimizer updates, reverse
chains, objective changes, or deployment into production.

Let independent Bernoulli edges B have probabilities p, and let
`F(B)=(c(B)-c_ref)^2/(N-1)^2`, including isolated nodes in c(B). For each legal
undirected slot e, define `D_e(B_-e)=F(B_-e,1)-F(B_-e,0)`. Directly differentiating
the finite product distribution gives

```
d E[F(B)] / d p_e = E[D_e(B_-e)]
d E[F(B)] / d z_e = p_e(1-p_e) E[D_e(B_-e)],  p_e=sigmoid(z_e).
```

This is the conditional expectation of the single-sample score-function term
`F(B)(B_e-p_e)` given B_-e. It has no larger **per-coordinate variance than that
unbaselined term**. There is no blanket variance ordering against LOO with a
baseline, or for their shared-parameter vector sum: different coordinates
condition on different sigma-fields, and cross-covariances matter. The finite
estimate remains noisy, may be zero, and is not a guarantee of protection.

We do not rerun connectivity once per edge. A bridge/component traversal on each
saved hard graph supplies all differences:

* If e is absent and joins two different components, writing k=c(B),
  `D_e=[1-2(k-c_ref)]/(N-1)^2`.
* If e is present and a bridge, `D_e=[-1-2(k-c_ref)]/(N-1)^2`.
* Otherwise `D_e=0`: toggling that single edge cannot change component count.

The formulas agree because D_e depends on the graph with e omitted, not on the
sampled value of B_e. For each sample, graph traversal is O(N+E); assigning
candidate differences costs O(M), with M the existing legal query slots. This
is an evaluation/gradient estimator, not a graph editing sampler. Fixed zero/
one probabilities have zero logit derivative, as required. No logit clipping is
used. Original reference edges are used only for per-group reporting; the score
uses the original reference component count, not edge identity.

Each state's estimate and original LOO estimate share the exact old hard samples.
All 4 failed training batches and 48 before/after evaluation batches are included;
there are 1600 saved hard graphs in total. The six old evaluation times are
1/75/250/550/800/1000. These are development inputs, not fresh confirmation data.

For each saved before/after probability pair, we also report
`mean(D_before).T @ (p_after-p_before)` and the paired sampled change in F. This
is a local approximation **in probability coordinates**, not the model-parameter
Jacobian or actual AdamW directional derivative. The finite-product expected risk
is smooth, but contains edge interactions; the first-order approximation need
not describe a large simultaneous change. Sampling standard errors are exported
as diagnostics, not confidence in the estimator being unbiased for a finite
observed batch, nor significance tests across training seeds.

The small test enumerates all 64 graphs on four nodes, verifies 768 independent
SciPy component-count toggle differences, and checks exact expected derivatives
and variance against the unbaselined score-function estimator. No neural model
is imported or constructed; PyTorch only loads CPU saved-input containers.

Run from repository root using the existing sparse_block environment:

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5s 190s \
python research_branches/component_feedback_cpu_20261110/run.py
```

The script sets a 180-second internal limit and refuses an existing output
directory. For independent reproduction select an isolated result directory;
never remove the historical output to rerun. Expected working memory is under
2 GiB (measured peak is recorded), CPU threads one, GPU zero. Reproduction source
hashes and original input hashes are in the output protocol. The result does not
authorize follow-up training.
