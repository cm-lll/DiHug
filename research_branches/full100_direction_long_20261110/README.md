# Fixed 100-update CPU toy continuation

User approved extending the two frozen direction rules to 100 updates, not real
graph training. Restore six complete update20 states; retain old input prefix and
freeze 80 new IID batches using SeedSequence([2026111052, initialization]). Shared
inputs, K8, unchanged hard risks and T100 posterior; no weight/LR search.

Common direction supplies its per-step actual displacement norm; arithmetic mean
direction receives the same budget and original decay once. Mean's own Adam
moments/step are saved but not used for step size. This remains a controlled
mechanism comparison, not a deployable optimizer recommendation.

Readouts raw/EMA 20 (reused), 50 and 100. Each new readout has exact full100 terminal
probabilities and 1024 paired paths. Report all costs, types and diversity; no
best checkpoint selection. No new CE arm: old CE20 cannot be compared as an equal
training-budget baseline for new 50/100 results.

One CPU compute thread, 2GiB RSS, worker300s/watch360s, GPU0. Six additional
save/reload tests at update50. Independent NumPy/full-kernel audit. Stop on failure
or bound; no automatic follow-on run.

```
python research_branches/full100_direction_long_20261110/watch.py
python research_branches/full100_direction_long_20261110/audit.py
```

Output directory must not exist for the worker. Prior results remain read-only.
