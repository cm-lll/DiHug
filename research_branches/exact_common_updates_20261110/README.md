# Exact-feedback common-gradient diagnostic

Only the combination of the five existing parameter gradients changes. Their
estimators remain analytic count/present-triangle/absent-triangle risks plus K64
LOO joint-degree/component risks. The mean-gradient four-update baseline is
read from `results/exact_feedback_updates_20261110`, not rerun.

With parameter gradients g_i and b=mean(g_i), the existing `project_mean` solves
min_v 0.5 ||v-b||² subject to g_i dot v >= 0. It operates on the parameter-space
Gram matrix, not on edge-logit gradients. The candidate gradient is v; the
intended gradient-descent displacement is -v. AdamW receives v with its original
history, preconditioner and decay. Thus actual displacement is NOT assumed to
remain a common descent direction. Every g_i dot actual displacement is logged.

The scalar five-risk mean is a reference objective/vector; the projected update
need not be the gradient of that scalar. This is not adaptive loss weight tuning
by validation, nor is it a finite-step safety guarantee. No new probe, accept/
reject, LR adjustment, clipping, or optimizer-state reset is used. Estimated-zero
gradients remain in the system. An exact zero combined gradient still lets AdamW
and EMA advance; it is not a rollback. The two sampled risk constraints apply to
the supplied estimates, not the unknown exact gradients.

Four historical A raw u2000 states, four updates each: 16 temporary updates,
40 forwards, 84 VJPs including four mean-linearity checks. Full parent state,
historical input hashes and auxiliary uniforms match the old baseline. Final raw
evaluation uses the same six development inputs; these are not independent new
confirmation inputs. Same K256 uniforms are used for the remaining two risks.
No reverse diffusion chain or formal training is run.

From repo root in the existing sparse_block environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_common_updates_20261110/prepare.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_common_updates_20261110/watch.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_common_updates_20261110/evaluate_cpu.py
```

Preparation refuses an existing output directory. Reproduction requires an
isolated copy with registered local assets; never remove historical results to
rerun. The watcher uses only an idle GPU0, one model/process, and stops this
ticket's children at the declared wall/GPU budget. No RSS cutoff is imposed.
The baseline/production sources remain unchanged. New sources are bound by SHA256.
