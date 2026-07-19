# Canonical-Ordered GPU Candidate Backend

## Scope

The `canonical_ordered` backend accelerates the frozen relocation/typed-2-switch
sampler without changing its transition contract. Graph adjacency, family edge
tables, relation degrees, candidate construction, compact features, listwise
scoring, selection, and mutation remain on CUDA.

For each transition it constructs an oversized ordered proposal stream and
accepts the first legal unique edits. The role phase preserves the canonical
targeted-relocation/random-mix split. The closure phase samples typed open wedges
directly before constructing degree-preserving 2-switches. It does not rank an
oversized random pool by utility, which was the semantic mismatch in the older
`legacy_pool` GPU backend.

This is distributional parity, not bitwise trajectory identity: NumPy and CUDA
use different random-number streams. Family counts and schema invariants are
exact; candidate action/family/utility distributions and final graph metrics are
validated statistically.

## Gates

- Candidate distribution gates pass on IMDB C16 and PubMed C16 (128 batches).
- Six CUDA state/candidate tests pass, including commit and CPU feature parity.
- IMDB C64, four seeds: GPU `612.25 +/- 23.31` triangles and
  `0.42270 +/- 0.00288` clustering; CPU canonical `621.50 +/- 25.65` and
  `0.43071 +/- 0.00605`.
- PubMed C16 context, four seeds: GPU `62754.5 +/- 1826.6` triangles.
- All eight full runs preserve every family count.
- Single-process 10k timing: IMDB `1060.6 -> 185.6 s` (5.71x), PubMed
  `429.0 -> 175.2 s` (2.45x). Peak reserved CUDA memory is 124 MiB.

Use `--sampling-backend gpu --gpu-candidate-mode canonical_ordered`. The older
`legacy_pool` mode remains available only for backward-compatible ablations.

## Batched optional context

`sample_gpu_role_guided_edit_diffusion_batched` reuses the same resident graph
state and canonical candidate backend for several independent generations. It
batches the MLP forward and, when context is enabled, constructs a
block-diagonal typed sparse graph for one shared HGT forward. There are no
cross-chain edges or messages. Per-chain candidate streams, random generators,
argmax decisions, and indexed mutations are unchanged.

The formal CLI reconstructs node types, family domains, marginal counts, and
anonymous role conditions from the self-contained context checkpoint. It does
not load a clean dataset during generation. `--evaluation-dataset-dir` is an
explicit optional post-generation reporting hook and is absent by default.

The mutable relation-degree table remains integer-valued. Its squared-error
energy and per-edit gain are accumulated in float64, then cast to the scorer
dtype only when fused with neural logits. This prevents phase-transition and
final-audit drift on large graphs such as ACM without relaxing structural
invariant checks.

On the IMDB formal smoke, two context-enabled chains produce exactly the same
final edge sets and action counts as two independent one-chain runs. In the
switch-refinement phase, batching two chains increases aggregate throughput
from `31.3--31.9` to `39.8` chain-steps/s (about `1.26x`) while peak reserved
memory rises from 128 MiB to 157 MiB. Batches of four and eight chains reach
`51.1` and `56.1` chain-steps/s (`1.62x` and `1.77x` over the mean single-chain
rate), with 224 MiB and 432 MiB peak reserved memory. End-to-end scaling is
lower than the isolated HGT-forward speedup because proposal selection and
ordered mutation deliberately remain chain-local. This optimization is most
useful for multi-seed generation and is independent of whether a dataset
passes the context-quality gate; legacy MLP scoring is batched by the same API.
