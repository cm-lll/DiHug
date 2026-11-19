# Actual AdamW displacement calibration: bounded toy check

This is CPU float64 arithmetic on a 4-node, 9-parameter model, not production
training. Use the bound real optimizer settings lr=2e-4, betas=(.9,.999), eps=1e-8,
decay=1e-12; CPU foreach/fused=False, while real GPU records are None. EMA=.99.
No inherited real checkpoint moments: all toy arms start fresh at the same initial
weights, so this does not diagnose historical trained Adam states.

Three old initializations, same 20x8 noised inputs, original 100-time kernel.
Three matched arms: CE AdamW, hard AdamW, hard AdamW actual-displacement-scaled.
One scalar is frozen before training using 72 independent fresh-state AdamW
proposal norm ratios (CE / hard), median clipped to [1,10]. No raw gradient scaling.
No automatic continuation or larger scalar if calibration offers no acceleration.

The candidate advances m/v/step once using the original hard gradient, applies
old + s*(native_proposal-old), then advances EMA once. Decoupled decay is included
in this full displacement. On a same-state step this equals scaling that step's
LR with the same gradient; subsequent gradients/moments naturally change. It is
not scaling the loss or a promise of common descent under Adam preconditioning.

Native torch.optim.AdamW is checked against independent NumPy equations. At step10
save/load and one-next-step comparison checks parameters, moments, EMA, cursor and
CPU RNG. Precomputed input bank is hash-bound; no CUDA replay claim is made.
Full100 chains evaluated raw at10/20, EMA at20; exact distribution plus1024 paired
paths each. Fixed noised evaluation uses the prior96-input bank, not untouched data.

One CPU thread, 2GiB RSS, worker300s/supervisor360s. No production source changes.
```
python research_branches/full100_adam_scale_20261110/watch.py
python research_branches/full100_adam_scale_20261110/audit.py
```
