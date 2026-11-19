# 100-step finite typed-diffusion check

Approved bounded CPU toy investigation. Four nodes, six legal pairs, 64 possible
graphs, nine shared arithmetic predictor parameters; NOT the production O/A
network. Existing untrained toy states are reused. Original three-point cumulative
retention is interpolated in normalized time to 100 steps, and both new toy arms
are trained on that same grid. No production model or dataset is changed.

Six toy tasks (three initializations, CE / existing hard five-objective cone
combination), 20 SGD updates each, eight paired IID time/noise inputs. Hard outputs
are exactly integrated, not approximated by a finite graph sample. No edge-count
constraint, new loss, weight search, optimizer change to production, or stopping
based on quality. Learning rate .2 is inherited from the old CPU toy protocol.

Each initial and endpoint predictor is evaluated through all 100 reverse steps,
with its own generated states. Exact probability propagation and 1024 paired
sampled paths agree. The known-X0 oracle is only a mathematical test. Actual
inference receives current state and time, not the reference graph.

Outputs and all costs/limitations: results/full100_toy_20261110/REPORT_ZH.md.
Original source/data bindings: protocol.json. The simulation is complete; do not
restart its watchdog or overwrite existing outputs.

Commands used:

```
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/full100_toy_20261110/watch.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/full100_toy_20261110/audit_summary.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/full100_toy_20261110/plot.py
```

Worker bound 180 seconds, independent supervisor 240 seconds / 2 GiB RSS, one CPU
compute thread. CPU only. These toy results do not establish production AdamW,
EMA, finite hard-sample gradient, real-graph convergence, or final graph quality.
