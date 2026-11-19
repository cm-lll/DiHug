# Matched temporary updates: exact versus sampled hard feedback

This bounded diagnostic uses four historical A late-CN raw u2000 checkpoints.
Two independent arms take four historical main inputs each. Both optimize the
equal arithmetic mean of the same five hard structure risks; there is no CE
training term, MGDA, cone projection, acceptance test, clipping, LR search, or
reverse generation. It tests an estimator change, not a production recipe.

- `sampled`: K64 independent hard clean draws, stable leave-one-out gradient for all five risks.
- `hybrid`: analytic expected count/present-triangle/absent-triangle squared risk gradients, K64 LOO for joint-degree/component risks.

The existing parent AdamW moments, LR, decay, and EMA are preserved, so this is
a CE-warm-start local experiment. Auxiliary random streams are separate and
paired; when arm probabilities diverge, paired uniforms do not imply identical
hard graphs. Every main input is checked against the old historical hash.

`prepare.py` binds source/assets and refuses an existing output directory.
`watch.py` checks GPU0 is idle, launches one model per process, and enforces the
840s science stop and 900s cumulative occupied-GPU budget without an RSS cutoff.
`worker.py` uses existing training transactions, preserves full state, checks
exact save/load plus next input identity, and records ordinary CUDA recomputation
differences without claiming bitwise continuation.
`evaluate_cpu.py` evaluates the saved raw predictions, after GPU release, on six
new fixed input states. Exact three risks plus K256 shared-uniform estimates of
the other two are reported, including paired Monte Carlo standard errors. These
are conditional clean predictions, not reverse-chain generated graphs.

Commands from repository root in the existing sparse_block environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_feedback_updates_20261110/prepare.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_feedback_updates_20261110/watch.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_feedback_updates_20261110/evaluate_cpu.py
```

Use an isolated copy and an absent output directory for reproduction; never
delete current results to rerun. GPU commands require the explicitly available
host device. Historical assets are private/local, with paths and hashes in
`results/exact_feedback_updates_20261110/bindings.json`. No dependencies installed.
