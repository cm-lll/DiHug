# Cached real probability hard-feedback check

No model imports/calls, no training, no reverse diffusion, GPU 0. Uses four
historical A late-CN raw u2000 exports, six fixed forward-corrupted inputs each.
K64 x four independent repeats. Frozen cost code is imported from the existing
CPU HardScore; only logit-margin gradients are estimated, not parameter gradients.

Run from repository root, in an existing sparse_block environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python research_branches/real_hard_feedback_cost_20261110/watch.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python research_branches/real_hard_feedback_cost_20261110/summarize.py
```

The worker refuses to overwrite its result directory or log. For reproduction,
use an isolated copy with a distinct output path and preserve original assets.
Bounded worker: 540 seconds; supervisor: 600 seconds cumulative computation
including recorded failed worker attempt, RSS 2 GiB; one CPU compute thread.
Summarizer is read-only with respect to inputs/samples, produces tables and an
analytic count-gradient reference, not an alternative scientific experiment.

Outputs: `results/real_hard_feedback_cost_20261110/REPORT_ZH.md` and protocol,
assets, bindings, per-sample/per-repeat tables, audits, worker/supervisor receipts.
First export failure and all affected source are preserved separately in
`results/real_hard_feedback_cost_20261110_attempt0_failed/`.
Engineering fixes do not change the frozen scientific rules or sampling seeds.
