# One-step response: same input versus other noise inputs

CORE diagnostic, user approved the preceding bounded next step. Reuse four A
raw u2000 complete checkpoints and the last ticket's unchanged displacement
projection. Each model makes exactly one temporary AdamW proposal and commits
the same corrected-displacement rule. No line search, extra gate, objective
change, learning-rate change, full reverse generation, or long training.

Main input is the historical update2001, t605/675/151/148 for IMDB0411/0412 and
PubMed0421/0422. Before prediction is the actual training forward; after uses
the identical input and model train mode under no_grad. All diagnostics preserve
the committed parameters, Adam, EMA, counters and complete RNG. Six other
development inputs are read in eval mode, with cached original baselines reused.
Do not compare train/eval across inputs as a pure noise-time causal contrast.

Three risks have exact expectations; joint-degree and components use paired
K256 evaluations independent of the K64 gradient sample on the training input.
Other inputs reuse original K256 uniform seeds. First-order slope G*d is compared
with actual finite risk difference on the same input. The remainder can include
nonlinear/active-region/numerical effects; it is not automatically a Hessian
curvature measurement or proof that lowering LR solves training. Sampled risks
also have gradient and evaluation sampling uncertainty.

Four temporary updates, 32 forwards and 20 VJPs. CPU projection/state test is
reused with verified source hashes; no redundant GPU acceptance panel. One idle
2080Ti, science420s, delivery600s from supervisor, <=.15GPU-hour including load,
save and occupied waiting; no artificial RSS limit. Budget is an upper bound,
not an obligation to keep running. End and release after this panel.

Run sparse_block Python with one CPU thread: prepare.py, watch.py, CPU-only
evaluate_cpu.py. The prelaunch binding refresh records the final evaluation
script/README after the small CE-denominator audit; use the original input's
ce_weight, never infer a different CE denominator. All raw files preserved.
