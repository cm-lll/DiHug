# Actual-AdamW direction projection: CPU-only toy

The user authorized one bounded followup, not real graph training. Keep existing
five hard risks, combined input gradient, production AdamW settings and full100
kernel. Project full native Adam displacement onto Gd<=0, using existing risk
gradients. No extra protective probes, nonlinear acceptance, rollback, weight or LR
search. The existing hard-gradient cone combination remains: only actual committed
displacement changes. Solver failures stop; a mathematically zero projection does
not roll back Adam moments/step. EMA advances once after parameter commit.

Three old 4-node/9-parameter initializations, three arms x20 updates, same K8 inputs.
Baseline replay must reproduce every saved old AdamW parameter/moment/EMA boundary.
Candidate and radial control have separate trajectories. The radial control uses
the candidate's actual norm at the same update index: its factor can exceed1 after
trajectories differ, so it is a norm-budget control, not necessarily always shrinkage.
Additionally every candidate step has exact same-state native/projected/equal-norm
radial/no-update shadows. Shadow forward evaluations do not decide acceptance.

No new models or production neural inference. Toy gradients integrate all64 outputs.
Raw10/20 andEMA20 full100 reverse chains are evaluated, exact probabilities plus
1024 paired random paths. No claims about finite-sample real gradient estimates.

Projection derives from min .5||d-delta||^2 s.t. Gd<=0, with rank-aware active-set
dual solving and independent scipy.optimize.nnls checks; no regularization. First
order constraint is only on current noised inputs, not finite steps/unseen states.
One CPU thread,2GiB,300s worker/360s watchdog.

```
python research_branches/full100_adam_direction_20261110/watch.py
python research_branches/full100_adam_direction_20261110/audit.py
```
