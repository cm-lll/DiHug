# Mean structural direction versus common-descent direction

CPU-only diagnostic authorized by the user's “可以做”. No production model,
GPU, new loss, multiplier search or long training. Original assets remain read-only.

Three archived toy initializations, 20 updates each, 8 identical IID inputs per
update. Replace `project_mean` direction by arithmetic mean gradient. Use the
archived common-direction arm's actual parameter displacement norm at each step;
keep original weight decay exactly once. This removes a direction constraint at
fixed displacement budget, not a comparison of deployable optimizers. AdamW
moments/step and EMA advance once; moment magnitude is not used to change budget.
Same-state shadows isolate directions on all 60 old pre-update states.

Full original 100-step posterior evaluation at raw10/raw20/EMA20, exact 64-state
terminal distribution and 1024 paired paths each. Reuse old common-direction,
matched ordinary AdamW and CE results; no new baseline fitting. All are development
evidence on a four-node reference, not IMDB/PubMed or generalization results.

Bounds: one CPU compute thread, 2GiB RSS, worker 300s/watchdog 360s; GPU zero.
Stop on zero mean direction, incompatible norm/decay, numerical or binding error.

Run once from repository root (output must not exist):

```
python research_branches/full100_mean_direction_20261110/watch.py
python research_branches/full100_mean_direction_20261110/audit.py
```

Evaluation: report all five risks and chain metrics, not just the average.
An improvement in average with a worsened triangle/component term is a tradeoff,
not a uniformly better or stable method. No automatic follow-on experiments.
