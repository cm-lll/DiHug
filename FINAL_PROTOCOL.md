# Frozen Paper Protocol

This repository is frozen to the target-free, GPU-resident role-guided edit
pipeline for the paper experiments. Historical HSD/GQER/controller experiment
configs have been removed from the active Hydra config search path so they
cannot be launched accidentally.

## Active method

- Initialization: independent family Binomial/Bernoulli prior.
- Denoising: learned role-guided relocation and typed switch edits.
- Optional frozen context residual: selected and reported as a paired ablation;
  it must not read clean adjacency or evaluation targets at sampling time.
- GPU path: resident graph state, streamed proposal pools, and fused Triton
  typed-closure scoring.

The unconditional paper main protocol uses `FAMILY_COUNT_MODE=binomial`.
`FAMILY_COUNT_MODE=fixed` conditions on the clean family edge counts and is
reported only as a conditional upper-bound/ablation, never as the main result.

## Canonical entry points

- `scripts/run_paper_final_context_main_8gpu.sh`
- `scripts/run_paper_cross_dataset_binomial_8gpu.sh`
- `scripts/run_paper_final_evidence_8gpu.sh`
- `scripts/run_pubmed_hgen40_nested_selection_8gpu.sh`
- `scripts/run_rooted_context_freeze_2gpu.sh`

`scripts/run_dblp_role_completion_8gpu.sh` is a step-budget diagnostic only;
its 150k result is not the paper main protocol.

## Do not revive

Do not launch or recreate experiment configs for GQER, family-set ranking,
absorbing posterior, block-global posterior, learned limiter/controller,
two-hop/query-gate, closure-bucket, node-closure, population quota, or old HSD
sampling sweeps. Their historical implementations remain recoverable through
Git history, but they are not active paper methods.

## Reproducibility rule

Every paper run must record the Git commit, dataset fingerprint, checkpoint
hash, seed range, initialization mode, and whether clean adjacency or metric
targets are accessible during sampling. Independently trained checkpoints from
different machines are reported separately rather than merged into one mean.
