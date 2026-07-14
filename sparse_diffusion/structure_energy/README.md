# Typed Degree-Conditioned Graphlet Energy

This package is a clean feasibility path beside the legacy diffusion system.
It deliberately does not call the old block sampler, controllers, GQER, or
post-hoc target-metric realizer.

## State Space

Training extracts each node's complete relation-degree vector and permutes
those vectors only within node type. For every relation family, the degree
multiset is therefore unchanged. A permuted copy of the training graph is a
constructive witness that the resulting state space is nonempty.

Generation never receives the training graph. The checkpoint contains only:

- node type counts and relation-family endpoint types;
- the multiset of relation-degree vectors within each node type;
- an automatically selected typed graphlet dictionary and its normalization;
- frozen linear energy weights.

It contains no edge list or original node identifier.

## Graphlet Model

The automatic dictionary currently covers all observed typed three-node
patterns rather than a hand-written metapath list:

- open typed wedges;
- typed triangles, including repeated node types;
- typed rooted triangles stratified by anonymous degree bucket;
- family-specific endpoint degree mixing.

Gate 2 asks only whether these statistics distinguish the real graph from a
degree- and family-preserving null. Pairwise NCE is used for that diagnostic;
its classifier weights are not treated as final ERGM parameters.

Gate 3 fits the fixed-degree maximum-entropy model with persistent MCMC. For a
frozen parameter vector, the negative log likelihood is convex and its
gradient is the model expected statistic minus the real statistic. Each
parameter round now keeps the energy frozen and extends the persistent chains
until at least 90% of selected statistics pass ESS, split-Rhat, and
Monte-Carlo drift tests. A diagonally preconditioned Robbins-Monro update uses
a diminishing step size; a failed expectation diagnostic skips the update
instead of pushing noisy gradients into the weight boundary.

Fresh null and energy chains are also extended adaptively. Gate summaries use
only the latest diagnostic window, and report ordinary Rhat, split-Rhat, ESS,
and early-to-late drift. In particular, ordinary Rhat near one is not accepted
as convergence when all chains share the same trend.

## Sampler Guarantee

Every proposal is either a symmetric, family-preserving 2-switch or a
symmetric Curveball neighborhood trade inside one bipartite family. Both keep
the graph simple and exactly preserve every node's degree in every relation
family. With frozen energy and Metropolis acceptance, the chain satisfies
detailed balance for the typed graphlet energy distribution on that state
space.

This guarantees the invariant state space and stationary distribution. It does
not claim that a finite chain has mixed; IAT, ESS, and multi-chain R-hat are
reported for that reason.

An optional replica-exchange sampler runs the same frozen energy at a ladder of
inverse temperatures. Its local kernels and exchange ratio are exact, so the
beta=1 replica has the same target marginal while gaining non-local paths
between structural modes.

## Commands

Run the training-time gates:

```bash
python scripts/run_structure_energy_gates.py \
  --output-dir results/imdb_structure_energy_seed0 \
  --seed 0
```

Generate from the saved checkpoint without a dataset argument:

```bash
python scripts/sample_structure_energy_checkpoint.py \
  --checkpoint results/imdb_structure_energy_seed0/structure_energy_checkpoint.json \
  --output-dir results/imdb_structure_energy_sample_seed100 \
  --seed 100
```

Use exact replica exchange when the ordinary chain has poor ESS:

```bash
python scripts/sample_structure_energy_tempered.py \
  --checkpoint results/imdb_structure_energy_seed0/structure_energy_checkpoint.json \
  --output-dir results/imdb_structure_energy_tempered_seed100 \
  --seed 100
```

The diffusion model may later provide only an initializer. A learned
state-dependent proposal is excluded from this version because it would need
an explicit Hastings correction to preserve the same target distribution.
