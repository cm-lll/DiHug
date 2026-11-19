# Actual AdamW displacement coordination: bounded development test

Track CORE, single-reference heterogeneous graph-state diffusion. User approved
the next small test, not formal training or full reverse chains.

Four frozen A raw u2000 models, four original inputs each. Keep five hard-graph
risks, exact moments for count/present/absent triangles and K64 stable LOO for
joint degrees/components. Mean parameter gradient goes into unchanged AdamW.
Read its actual proposal d (including momentum, preconditioning and decay).
Solve min 0.5||x-d||² subject to Gx<=0, using rank-aware active-face enumeration.
Commit corrected FP32 parameters only after checking actual written slopes;
keep the proposal's Adam moments and step, then update EMA once. Zero parameter
displacement would still advance Adam/EMA: it is not a full rollback.

No additional acceptance probes, line search, learning-rate change, CE term,
gradient-cone projection, clipping, strict deterministic mode, or new metrics.
Solver failure or written-direction violation stops the affected experiment;
it cannot be represented as an accepted zero step or an exact-resume checkpoint.
The existing training boundary prohibits checkpoints after partial optimizer
failure. CPU checks use an independent primal solver and a real toy AdamW step.

Reuse both old mean and common-gradient endpoints. Evaluate only the four new
raw endpoints on the same six development Xt and K256 shared uniforms. These
are conditional clean-graph risks, not full-chain generation quality. Two risks
remain estimated; first-order constraints do not guarantee finite-step decrease.
Ordinary CUDA trajectory equality remains uncertified; state loading is checked.

Limits: 16 temporary updates, 40 forwards, 84 VJPs; one idle2080Ti; 20min wall,
840s science stop, <=.25GPU-hour including loading/saving. Independent watch.py
accounts allocation-to-release, no artificial RSS cap. Expected cost from last
ticket approximately 4.5min supervisor / .057GPU-hour plus CPU audit.

Run with sparse_block Python, single CPU threads:
`prepare.py`, `watch.py`, then CPU-only `evaluate_cpu.py`.
All original assets remain unchanged; bindings.json and protocol.json identify
exact assets/source. Evaluation script added after CPU preparation is bound in
the final prelaunch binding refresh. Results stop after this development panel.
