# 6000-update paired correction-training panel

Independent review branch. Scientific source is fixed at
`3a8d55234ccbd13859aed7f2d76fe17c5d0d4cf2`; preparation baseline is
`3bcbbd91545c24b32961bde52ac738262b9aa99d`. Main and previous versions are preserved.

This is the authorized 34-node development experiment, **not** IMDB/PubMed
large-graph training. Two arms (`normal2`, `robust`), three paired training seeds,
6000 effective updates each: 36000 optimizer updates, 72000 training query
forwards. All initial parameters are copied and checked tensor by tensor;
optimizers are independent and continue across milestones.

Both arms average two original R/S query CE estimates. The control sees two
ordinary forward-noise inputs; the candidate replaces only its second input
with a traceable artificial add/delete corruption, in both target conditions.
This adds a robustness objective; it is not the unchanged diffusion CE
objective. No hard roles, probability guidance, quotas or graph repairs.

Milestone checkpoints: 1000/2000/3000/4000/5000/6000, with model, optimizer,
RNG, counters and bindings. Raw full T64/R4 generation is fixed at
1000/3000/6000: eight paired sampling seeds per target and train seed, 288
chains in total. Validation is frozen at 0/250/500 and every500 thereafter.
No checkpoint selection or output filtering.

The unique activation ledger limits the entire panel to28800 seconds. Three
single-thread CPU seed jobs share its original deadline. The GPU numeric
gate failed its predeclared gradient tolerance (5 elements, maximum2.519e-6
versus2e-6); tolerance was not changed and no scientific GPU run is claimed.
The GPU gate receipts are included. Active STOP is deferred to the end of a
complete two-arm optimizer-update pair; hard timeout still interrupts.

`RESULTS.md` is the readable result. `evidence/learning_run01` contains all
raw graphs, per-update logs, fixed validation, pressure probabilities,
sampling RNG/query digests, paired summaries, cost/audit/exit receipts and
weight hashes. Model/optimizer tensor files remain in local persistent
storage; `local_checkpoint_inventory.json` gives every hash and byte size.
They are not required for read-only review of the published results.

In the GitHub results directory, reproduce read-only aggregation (never launches learning):

```bash
python analyze.py --run evidence/learning_run01 --output reproduced_analysis
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/dihug_mpl \
  python plot_curves.py --analysis reproduced_analysis
```

After extracting the ZIP, the identical evidence folder is named `results`:

```bash
sha256sum -c SHA256SUMS
python analyze.py --run results/learning_run01 --output reproduced_analysis
python role_robust_training_v1/run.py --help
```

The outer GitHub results directory (which contains both the ZIP and verifier)
supports full independent ZIP verification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  python verify_release.py
```

This runs the exact135-test CPU collection and public help/binding checks;
it does not start this scientific panel again. The original launch command
and environment are archived in activation/config/source receipts. No
automatic follow-up experiment is authorized.
