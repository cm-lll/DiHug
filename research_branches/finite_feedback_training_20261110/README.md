# K64 finite hard-feedback training, CPU micrograph only

Three existing toy initializations, two direction rules, 100 updates from zero.
Same frozen 100x8 forward input bank as exact-feedback arms. Per input draw64
independent clean graphs and compute all five per-input LOO gradients, then mean
or existing project_mean. No CE or new loss. No real graph/network training.

Step norms are frozen to the exact common-direction arm's saved norms for both
finite arms. This isolates noisy direction feedback; it does not test an adaptive
magnitude rule driven by noisy gradients and is not a deployable optimizer.
Own estimated gradients advance Adam moments/step; decay once and EMA once.
Zero estimated direction means decay-only, explicitly logged, no exact fallback.

Exact gradients/risk are read-only diagnostics. They never select/reject an update.
Training output uniforms use SeedSequence([2026111056,seed]) shared across arms;
probabilities differ so sampled graphs can differ. Full RNG and cursor are saved.
Evaluation raw/EMA20/50/100: unchanged full T100 posterior, exact finite terminal
distribution plus1024 paired paths. No test-driven choice of checkpoint or K.

600 new toy updates,307200 training hard draws,12 recovery updates,36 readouts and
36864 CPU microchains. One hard-feedback stream per initialization; this is not
an independent sweep of gradient-estimator seeds. No new CE arm.

Single CPU compute thread,2GiB,worker300s/watch360s,GPU0. Existing exact results
reused. Independent complete sample/LOO/NNLS/state/EMA/RNG/kernel audit.

```
python research_branches/finite_feedback_training_20261110/watch.py
python research_branches/finite_feedback_training_20261110/audit.py
```

Output must not exist before worker. Original assets remain read-only. Initial
source compilation found and removed one extra list bracket before any experiment;
this build failure is recorded, not a scientific rerun or changed protocol.
