# Fixed update scale diagnostic (CPU toy only)

User authorized one scalar calibration after full100_toy, not production training.
Three existing 9-parameter untrained toy initializations, same 20 updates, eight IID
noised inputs/update, same five exact hard-graph risks and projected-mean direction.
Only change: one globally frozen positive displacement multiplier.

Before new training or terminal evaluation, compute 24 independent calibration
batches of eight inputs for each initialization at its untrained parameters.
Use the median of the 72 CE-gradient/projected-hard-gradient norm ratios,
bounded to [1,10]. Stop if any projected direction <=1e-12 rather than omit it.
The upper bound is a diagnostic resource/risk choice, not a theory-derived optimum.
All seeds use the same fixed scalar. CE is a calibration yardstick only, not an
auxiliary loss. No per-step clipping, loss acceptance checks, or coefficient search.

Replay original hard arm and require bitwise old final parameters; reuse CE/initial
and old hard terminal distributions. Evaluate new hard endpoints at updates 10/20,
original at 10, using exact full100 reverse-chain propagation and 1024 paired paths.
Every update additionally logs a separate 96-input fixed noised validation bank.
Calibration/evaluation streams are distinct; finite states may coincide, and the
reference and three historical initializations are development data, not unseen graphs.

One CPU thread, 180-second worker/240-second supervisor, 2GiB RSS, GPU=0.
Run with existing environment:

```
python research_branches/full100_scale_20261110/watch.py
python research_branches/full100_scale_20261110/audit.py
```

Positive scaling preserves first-order cone signs at the same parameters. It does
not preserve finite-step decreases, and following directions change with trajectory.
Plain SGD toy results do not establish equivalent behavior for production AdamW.
