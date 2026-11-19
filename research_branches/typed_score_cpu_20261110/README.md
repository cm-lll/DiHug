# Typed hard-graph score audit (CPU, 2026-11-10)

This is an isolated description/score audit, **not a production loss change**.
It reuses 52 saved batches / 1600 hard clean-prediction draws. It imports no
denoiser, performs no optimization, and generates no new real-data samples.
Controlled finite graphs are CPU mathematical tests, not learned generation.

## Frozen definitions

All pairs are unique unordered non-self pairs from the actual active query
inventory. Fixed-zero pairs are absent from support. Relations are uniquely
determined by each supported pair in these bound datasets; this implementation
does not claim support for parallel multi-relational edges.

For each node, `degree` is its vector of relation degrees. `local` records the
number of pairs of distinct neighbors, keyed by the unordered pair of their
incident relation IDs. Three separate channels distinguish closed wedges,
open wedges with a legal possible closing edge, and paths with an unsupported
closing edge. The third category is not interpreted as a missing triangle.
Node types are kept separate during matching. Features feasible in the full
legal inventory are retained even when zero in the reference.

For each type, local columns are divided by `max(1, reference-node RMS)`.
These scales are frozen from the reference, never fitted to generated graphs.
`J` is the unchanged previous Hungarian mean L1 joint-relation-degree distance
divided by N-1, averaged equally over node types.
`L` matches the local vector with mean scaled L1 cost.
`JL` uses one Hungarian assignment for the sum of degree and local costs.
The nonnegative gap `JL-J-L` measures the cost of requiring a shared assignment.
These are diagnostic costs; their relative scales are **not** a validated
training-weight rule. JL can still hide losses in individual feature groups.

Each node's component vector gives the number of OTHER reachable nodes of
each type, divided by the available inventory of other nodes of that type.
`C` is type-wise Hungarian mean L1 distance between these vectors. It captures
component size/type organization, not all topology inside components. If all
draws are connected, C is constant; batch LOO feedback can still vanish.

These deterministic per-graph scores depend only on the graph, bound inventory,
reference and frozen scales. They do not contain GDPO's batch-dependent
generated-self kernel term. For independently sampled graphs, the ordinary
score-function identity and independent leave-one-out baseline apply, subject
to differentiability of the sampling probabilities; no variance or descent
guarantee is implied. No network gradient was evaluated here.

## Checks and results

- Independent triple enumeration on 80 four-node graphs across two supports.
- Fixed exhaustive 32768 six-node graphs: deterministic witnesses for degree /
  triangle-total ambiguity and degree / triangle-marginal ambiguity.
- Typed C4+C8 versus C6+C6: component count/degree/triangle scores tie, C differs.
- Cube versus Möbius ladder: all proposed costs tie, but 4-cycle counts differ.
  This deliberately retained limitation prevents claims of complete structure.
- Real reference self-score and type-preserving relabeling checks; old J matches
  all saved costs. 68 bound files unchanged. No graph-isomorphism completeness
  or learned quality claims.

The first attempt imposed an address-space cap, which prevented PyTorch's shared
library mapping while RSS was only 149 MiB. The successful attempt uses an
independent 2 GiB resident-memory / 297 s supervisor. Original failure records
are retained. Scientific definitions were not changed. The second CPU pass
only decomposes the **same** JL assignments; it does not refit or revise scores.

## Reproduce into a new output directory

From repository root, using the existing environment (no installation):

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 AUDIT_RUN_SUFFIX=_reproduce AUDIT_WALL_SECONDS=297 \
 python research_branches/typed_score_cpu_20261110/watch.py
```

The audit refuses to overwrite an existing output directory. `decompose.py` and
`summarize.py` read the completed `_attempt2` directory for this run. They are
reporting scripts, not production or training entry points.

Actual deliverables: `results/typed_score_cpu_20261110_attempt2/`; first failure:
`results/typed_score_cpu_20261110/`. Main 41.30 s, attribution 33.52 s, failed
loader/tests 2.01 s. Highest independently sampled main RSS 414.57 MiB. This
excludes authoring/reporting time and is not GPU training throughput. GPU 0.
