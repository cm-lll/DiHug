# One scalar preserving the existing structural direction

CPU-only 4-node/9-parameter diagnostic, not production AdamW replacement.
Use the existing five-risk combined gradient g. Run native AdamW to update m/v/step
once and get its complete displacement. Separate decoupled decay, retain only the
norm of the adaptive part, and commit `-norm(adaptive)*g/norm(g) + decay`.
EMA advances once. Moment directions are deliberately not used; their effect on
the norm remains. No new loss, no step search, no additional protective probes.
Zero g commits decay only and still advances moments. Numerical failure stops.

Same three old initializations,20 updates,K8 input bank,production numerical Adam
settings as prior toy,CPU float64. Original AdamW full replay and two new arms:
scalar direction vs ordinary Adam direction with the exact same per-step norm
budget. Separate trajectories/optimizer states, plus same-state local shadows.
The norm control may need >1 factor once trajectories diverge; record, never cap.

Reuse old raw/EMA baselines; new raw10/20 andEMA20 exact full100 distributions and
1024 paired sampled paths. Fixed96-input development bank for non-decision reports.
This is a new optimizer rule, not plain AdamW, and still not a finite-step or final
graph guarantee. Same-reference toy data do not prove real graph generalization.

One CPU thread,2GiB,300s worker/360s watchdog.
```
python research_branches/full100_adam_scalar_20261110/watch.py
python research_branches/full100_adam_scalar_20261110/audit.py
```
