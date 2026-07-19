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
