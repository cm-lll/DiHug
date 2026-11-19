# Saved-probability density / arrangement diagnostic

Run in the existing sparse_block environment, one CPU thread and CUDA hidden:

```sh
timeout 600s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/density_arrangement_audit_20261110/run.py
```

The runner refuses to overwrite its output directory. Existing model outputs are read, never regenerated. No network is instantiated, no autograd is used, no real hard graph is sampled. Sixty-four four-node states are enumerated for an independent numerical check.

For each relation f, transform interior probabilities by
`q_e = sigmoid(logit(p_e) + b_f)` and solve `sum(q_e) = target_f`.
Exact zero/one probabilities are preserved, not clipped. Unsupported changes of endpoint support or unreachable targets are reported, not silently regularized.

This preserves within-relation pairwise log-odds differences and rankings. More strongly, for independent Bernoulli edges conditioned on their relation counts m_f, a graph's probability is proportional to the product of its selected odds. A common shift multiplies every eligible graph's unnormalized probability by the same `exp(sum_f b_f*m_f)`, leaving that conditional arrangement law unchanged. This does NOT fix the unconditional hard edge count, preserve each node's expected degree, or prove that the network can implement the shifted probabilities.

For parent p0 and changed p1, let F00 use p0, F11 use p1, F10 use p0 shifted to p1's expected counts, and F01 use p1 shifted to p0's expected counts. For any recorded risk F:

```
density = ((F10-F00) + (F11-F01))/2
arrangement = ((F01-F00) + (F11-F10))/2
interaction = F11-F10-F01+F00
density + arrangement = F11-F00
```

Both orders are retained because risk is nonlinear. This is an algebraic symmetric attribution under a chosen probability intervention, not a causal identification of network modules. The count distribution's variance can still change when expected counts match, so arrangement may also affect the count squared risk.

Exact Bernoulli expected squared risks include triangle variance and covariance between triangles sharing an edge; they are not just squared errors of expected triangle counts. NumPy formulas are checked against exhaustive micrograph sums and every cached real endpoint. Relation-level moments and typed-triangle means, variances and risks are retained. Joint-degree matching and connected components are not recomputed or inferred from this decomposition.

All four historical models, seven input states and three predeclared alpha endpoints are included. They are development diagnostics on conditional clean predictions, not complete reverse generation or independent replication. The older fixed-hard-count experiment used different update states and is background only. Nothing here selects a step size, retrains a model or changes the production sampler.
