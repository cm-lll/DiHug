# Exchangeable Typed Edit Diffusion

This package contains the graph-generation method used by the final IMDB and
PubMed experiments. It is independent of the legacy block sampler, GQER,
family-set loss, and sampler-side target-metric controllers.

## Generative Model

Generation has two learned stages:

1. An exchangeable joint relation-degree prior samples anonymous node roles
   within each node type. The prior stores mixture parameters and type/family
   schema, not training node IDs or a training edge list.
2. A learned edit-space reverse process scores legal typed graph edits. The
   complete training kernel contains birth, death, relocation, and 2-switch
   transitions. Each candidate is represented by its action, relation family,
   anonymous endpoint roles, diffusion time, and typed closure deltas.

The known forward process starts at a training graph and applies a
time-inhomogeneous mixture of the same edits. Birth/death is a Bernoulli
heat-bath update; relocation and switch are reversible typed edits. Training
records noisy states and learns to rank inverse or reconstruction-improving
edits with a listwise objective.

The strongest validated sampler is staged. It samples a graph from the
exchangeable prior and runs the count-preserving relocation/switch subset of
the learned reverse kernel as terminal motif refinement. This is an
edit-space discrete diffusion pipeline, but it is not the textbook
independent-edge D3PM sampler and should not be described as one.

## Inference Contract

The full-graph IMDB and PubMed samplers receive only:

- node type counts and the relation schema;
- learned exchangeable-prior parameters;
- learned reverse-kernel parameters;
- a random seed and fixed sampling budget.

They do not read the clean adjacency, exact clean degree rows, family edge
counts, graphlet target values, or evaluation metrics. The clean graph is
loaded only after generation to compute reported metrics.

The HGEN40 protocol is a separate protocol-conditioned evaluation. It applies
the public HGEN edge-cover view to an independently sampled full graph and is
conditioned on the public total edge budget. Hyperparameters are selected on
independent HGEN-style validation splits; the official held-out graph is used
only once for final evaluation. This setting must be labeled separately from
unconditional full-graph generation.

## Final Entry Points

Run the full-graph experiments:

```bash
bash scripts/run_imdb_exchangeable_compact_switch_main_8gpu.sh
bash scripts/run_pubmed_exchangeable_compact_switch_main_8gpu.sh
```

Run nested HGEN40 model selection and official evaluation:

```bash
bash scripts/run_pubmed_hgen40_nested_selection_8gpu.sh
```

Run the complete paper queue:

```bash
bash scripts/run_paper_final_queue.sh
```

## Retained Ablations

The package still includes the mixed-kernel, factorized-head, HGT-head,
fixed-degree switch, and graphlet-energy implementations required to reproduce
paper ablations. Their old one-off launchers are not part of the final runtime
surface. Git history retains removed GQER, family-set, absorbing-posterior, and
QPPR experiment configurations.

## Guarantees and Limits

Typed edits preserve graph simplicity and schema validity. Relocation and
2-switch preserve per-family edge counts; 2-switch additionally preserves
per-node relation degrees. The forward kernels and legal inverse operations
are explicit, so the learned process has a well-defined discrete Markov state
space.

These invariants do not guarantee that a finite learned reverse trajectory
matches every statistic of the data graph. Structural quality remains an
empirical result and must be reported over multiple seeds. In particular, the
current full PubMed result matches triangle count and transitivity closely but
still underestimates average clustering.
