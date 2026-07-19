# Candidate Clean-Edge Energy Gate (2026-07-19)

## Question

Can the Phase-2 role-conditioned HGT representation improve the frozen typed
edit MLP without replacing the validated relocation/switch kernel?

For a complete edit `a`, the tested residual is

```text
delta_edge(a) = mean(new-edge clean logits) - mean(old-edge clean logits)
score(a) = score_MLP(a) + lambda_action * normalized(delta_edge(a))
```

The current noisy graph is encoded once. All old/new edge slots from the free
candidate list are decoded in one batch. HGT, the legacy MLP, candidate
generator, and role plan are frozen. Only two non-negative calibration
coefficients are fitted on training trajectories. A zero coefficient is exactly
the legacy MLP.

Clean adjacency is used only to build training utilities and held-out gate
metrics. It is not available to the inference scorer.

## Held-out free-candidate gate

| Dataset | Expected utility gain | Selected utility gain | Edge/utility correlation | Decision |
|---|---:|---:|---:|---|
| IMDB | +0.0814 | +0.0625 | 0.2810 | pass |
| PubMed | +0.0936 | +0.1875 | 0.2672 | pass |
| DBLP | +0.0233 | +0.1250 | 0.1609 | pass |
| Syn-200 | -0.0001 | +0.0625 | 0.1875 | non-regression only |

All learned switch coefficients collapsed to zero. The useful HGT signal is
therefore relocation-specific; typed closure remains the legacy MLP's job.

## Paired 2k-step trajectory gate

The base and fused runs share seed 0, Binomial family-count initialization,
candidate count 16, role plan, edit budget, and all sampler parameters.

| Dataset | Metric | Base | HGT edge residual | Real |
|---|---|---:|---:|---:|
| IMDB | triangles | 548 | 514 | 655 |
| IMDB | clustering | 0.3867 | 0.3291 | 0.4689 |
| PubMed | triangles | 10,218 | 10,306 | 61,137 |
| PubMed | clustering | 0.0363 | 0.0374 | 0.3351 |
| Syn-200 | triangles | 297 | 312 | 261 |
| Syn-200 | clustering | 0.1235 | 0.1023 | 0.0839 |

DBLP sampling was not scored because the historical checkpoint uses
`link_Paper_Author` while the clean workspace graph schema uses
`link_Author_Paper`. The shallow candidate gate remains valid because it does
not instantiate the checkpoint marginal prior; no schema rewriting was used to
manufacture a sampling result.

## Decision

Do not advance this residual to 5k/full sampling or the frozen paper method.
The experiment establishes a narrower result:

1. HGT clean-edge representations contain complementary local relocation
   information on all three real datasets.
2. Local clean-edge utility is not a sufficient surrogate for long-horizon
   typed motif generation.
3. The residual changes the relocation trajectory and phase-transition state;
   this helps PubMed slightly but materially harms IMDB.

The validated legacy MLP and the previously frozen optional PubMed rooted
context remain the paper protocol. This gate is retained as reproducible
negative evidence, not a new component.
