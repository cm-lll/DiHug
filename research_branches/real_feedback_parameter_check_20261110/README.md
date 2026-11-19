# Cached hard-feedback parameter-gradient check

Read-only differentiation of four historical A late-CN raw-u2000 models at
t=1,250,800. Uses the exact K64 x four cached hard-sample batches from
`results/real_hard_feedback_cost_20261110`. No new sampling, optimizer steps,
EMA updates, reverse chains, or scientific configuration changes.

For each condition the 20 cached per-component margin gradients are mapped by
`torch.autograd.grad` to model parameters. Additional VJPs: analytic relation
count expectation, mean of the first five components for linearity, and one
fresh-forward repeat per model. Total completed panel: 16 forwards, 268 VJPs.
Network outputs must match cached probabilities within the predeclared 2e-6
identity tolerance. This is NOT the old optimizer replay tolerance and does
not certify bitwise GPU recomputation.

`run.py` uses the existing runtime restore and frozen-Xt query entry, reads
all state on CPU before restoring, and verifies model/Adam/EMA/RNG/counters
unchanged after each condition. Parameter group ranges are saved; shared
backbone/CN-context parameters are counted once. Outputs contain float64 Gram
matrices, which suffice for all reported comparisons without retaining hundreds
of megabytes of parameter vectors. Probabilities and cached draws are bound by
SHA256; no private checkpoints copied into outputs.

The first multi-model process hit the 4 GiB CPU RSS watchdog while switching
models, after completing IMDB0411. That process and source are preserved under
`results/real_feedback_parameter_check_20261110_attempt0_partial`. The remaining
models used separate processes via `resume_watch.py`, retaining the original
window and accumulated device time. Completed conditions were not repeated.

To reproduce one isolated model in an environment with an accessible idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DIHUG_MODEL_SEED=2026110411 DIHUG_DIAG_OUT=/tmp/dihug_vjp_reproduction_0411 \
python \
research_branches/real_feedback_parameter_check_20261110/run.py
```

Run a resource watchdog as for this ticket; do not overwrite original output
folders. `resume_watch.py` is specifically bound to this ticket's prior receipt
and original deadlines, and is not a fresh-run launcher. `summarize.py` reads
the aggregate condition list/Grams and runs on CPU only. Imported original
sources are checked against historical bindings; any future source drift must
be reviewed, not bypassed.
