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
- typed geometrically weighted edgewise shared partners (GWESP), stratified by
  relation family, endpoint degree roles, and shared-neighbor type;
- family-specific endpoint degree mixing.

GWESP is a curved-ERGM closure statistic: the first shared partner on an edge
has full weight and additional shared partners have geometrically diminishing
returns. It therefore represents typed distance-two support while discouraging
all triangles from collapsing onto a few hub edges.

Open-wedge counters are still computed for diagnostics, but are excluded from
the fitted energy by default. Under a fixed per-node relation-degree plan, the
number of typed neighbor pairs at each center is fixed, so open wedges are an
affine restatement of the corresponding closed-triangle statistics rather than
an independent constraint.

Gate 2 asks only whether these statistics distinguish the real graph from a
degree- and family-preserving null. Pairwise NCE is used for that diagnostic;
its classifier weights are not treated as final ERGM parameters.

Gate 3 fits the fixed-degree maximum-entropy model with persistent MCMC. For a
frozen parameter vector, the negative log likelihood is convex and its
gradient is the model expected statistic minus the real statistic. Each
parameter round now keeps the energy frozen and extends the persistent chains
until at least 90% of selected statistics pass ESS, split-Rhat, and
Monte-Carlo drift tests. A diagonally preconditioned Robbins-Monro update uses
a diminishing step size and a small trust radius. If a diagnostic fails, the
weights remain frozen and the next round accumulates more samples from the same
target distribution. Samples are cleared and warm-up is repeated only after a
successful parameter update.

The IMDB adaptive scripts use a damped natural-gradient update. For an
exponential-family graphlet energy, the Fisher matrix is the covariance of the
normalized sufficient statistics. The update solves this covariance system and
is scaled to satisfy a local KL trust region, with an Euclidean cap retained as
a fallback for covariance-null directions. The older diagonal update remains
available as an explicit ablation.

The formal adaptive scripts also use moment continuation. Instead of jumping
directly from the degree-preserving null expectation to a real-graph statistic
that can be hundreds of null standard deviations away, they track a geometric
sequence of intermediate moment targets. A stage advances only after the
frozen-chain expectation is both diagnostically valid and close to its current
target. The KL radius is reduced in early stages, so continuation changes the
distribution gradually rather than merely rescaling the same gradient.

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

Before the expensive fresh-chain validation, run the moment-fitting pilot. It
stops after optimization and reports whether enough diagnostic-approved updates
were made:

```bash
SEED=0 bash scripts/run_imdb_structure_energy_adaptive_f48_pilot.sh
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

## Switch-Space Diffusion

`switch_diffusion.py` implements a separate learned reverse process directly
on the degree-conditioned state space. Its forward kernel applies uniformly
sampled legal family-preserving 2-switches. Training records an exact inverse
switch together with negatives drawn from the same legal proposal kernel, and
fits a listwise classifier. This is sampled reverse-kernel estimation, not
independent edge classification.

Every forward and reverse action preserves graph simplicity, every family edge
count, and every node's degree in every relation family. Training uses
type-preserving permutations of the clean witness to avoid binding anonymous
roles to node IDs. A sampling checkpoint contains only anonymous role rows,
the automatically selected typed graphlet statistics, and learned scorer
parameters; sampling does not load the training adjacency.

Three sampling modes isolate failure causes:

- `random` measures the degree-conditioned switch null;
- `oracle` selects the candidate with the largest immediate reduction in the
  checkpoint statistic error and tests candidate-proposal coverage;
- `model` uses the learned reverse-kernel score.

The hard guarantee is membership in the feasible role state space after every
step. Finite-step graphlet recovery remains an optimization result, so reports
include validation top-k reverse accuracy, model-oracle agreement, graphlet
target error, and graph metrics along the trajectory.

Run the causal gate:

```bash
python scripts/run_switch_diffusion.py \
  --structure-checkpoint results/imdb_structure_energy_seed0/structure_energy_checkpoint.json \
  --output-dir results/imdb_switch_diffusion_seed0 \
  --trajectories 8 --forward-steps 2000 --record-every 20 \
  --train-candidates 32 --sample-steps 2000 --sample-candidates 64
```
