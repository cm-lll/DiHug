# Finite hard-graph feedback reliability

CPU-only fixed-state audit; no optimization, production denoiser or reverse chain.
Use all 3 toy seeds x 2 trajectories x raw20/50/100 states (18), each evaluated at
t=5/50/95 with 8 pre-frozen forward-noised inputs. The same inputs are shared
across weights; they are new noise of the same reference, not unseen graphs.

Per input draw K=4/16/64 independent clean hard graphs from its prediction. Use
existing unbiased per-input leave-one-out estimator for all five costs, averaged
over the 8 inputs. K arms are nested prefixes; 64 independent repetitions per
condition. This is not sampling more noise times or generating full chains.

Compare estimated mean/projected directions with exact enumeration gradients.
Report gradient error, cosine, zero feedback, true first-order component changes
and harm magnitudes (not just sign). Projection of unbiased estimates is nonlinear
and not unbiased; estimated constraints do not guarantee exact-risk constraints.
No actual parameters change, no claim of generation improvement.

Counts: 54 conditions, 10368 gradient-estimation batches, 1769472 unique tiny hard
graph draws. Reused K prefixes are not counted as extra independently drawn graphs.
Costs use a 64-state lookup: timing is a lower-cost toy implementation, not a real
graph scoring throughput estimate. One CPU thread/2GiB, worker300s/watch360s/GPU0.
Rule, streams and panel are frozen before sampling. No adaptive K or retuning.

```
python research_branches/finite_feedback_precision_20261110/watch.py
python research_branches/finite_feedback_precision_20261110/audit.py
```

Output must not exist for the worker. Independent audit rebuilds all 64 costs,
every saved sample and LOO gradient, checks exact K2 unbiased expectation and
independent NNLS projection on 162 representative batches. Stop after report.
