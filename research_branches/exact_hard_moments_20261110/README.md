# Exact expectations of existing hard count/triangle risks

CPU-only research implementation; not wired into production training. `moments.py` computes count and typed-triangle expected squared errors including variance/shared-edge covariance. `remaining.py` retains the original joint-degree and component scores. No objective weights or denominators change.

From the repository root, with the existing sparse_block environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_hard_moments_20261110/run.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/exact_hard_moments_20261110/audit_and_summarize.py
```

`run.py` deliberately refuses an existing output directory. To reproduce, use an isolated repository copy with the registered historical assets and an absent new results directory; do not delete original results. `audit_and_summarize.py` expects the completed main run and writes audit/summary files. All required historical paths/hashes are recorded under `results/exact_hard_moments_20261110/`. Dependencies are existing NumPy, SciPy and CPU PyTorch; no installation performed.

Real panel: four A raw2000 models, six cached states each, previous K64 x four repeats reused. No model calls, new real graph draws, training, or GPU. Full Chinese explanation and limitations: `results/exact_hard_moments_20261110/REPORT_ZH.md`.
