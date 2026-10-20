# Role allocation versus edge prediction: staged follow-up

User authorization 2026-10-20: continue toward a stage result; routine engineering
repairs delegated. Base b8e97cc. Old experiments and budgets remain closed.

Stage 1: CPU-only diagnostic replay, 300 seconds maximum, three archived
capacity-C final weights, both conditions, first pre-existing sampling seed
2026102031. Reproduce all 64 states and RNG receipts; inspect t=64,32,16,8,4,2,1.
These are six replayed existing samples, not additional quality observations.
Capture full frozen Xt, p0, posterior q, allocation, birth/death expectations.
No role argmax is ground-truth role identity. Decide the next modification only
after this evidence. No notebook/BW or large graph execution.

If low soft-target nodes remain overconnected in p0, the sole candidate is
adding soft target relation degrees and noise-aligned observed-degree residuals
to the existing nonlinear joint head. Full joint role embeddings stay.
No probability controller, degree loss, hard quota, graph repair or optimizer
change. A new learned candidate requires a separately frozen bounded panel.

## Diagnosis completed / sole candidate selected

Six archived chains reproduced all 384 state/RNG hashes. At t=1 extreme,
the five nonhub soft targets average 1.2 but predicted clean degree averages
1.850, 2.457, 2.466 across three trained seeds. Balanced soft targets are all
4 while predictions substantially track unequal Xt degrees. This supports an
explicit degree-input test, not a proof that allocation is always correct.

For anonymous soft allocation A, role vectors v, capacity N and original
retention a_t, define m_ir=sum_j A_ij v_jr and
b_ir(t)=a_t m_ir+(1-a_t)N_ir*pi_r. Extra node inputs are
[m_ir/N_ir, (d_ir(Xt)-b_ir(t))/N_ir], zero on N=0 dimensions.
No division by a_t or epsilon. A depends on Xt, so b is an informative
noise-aligned descriptor, not an unbiased estimate conditional on Xt.
Keep full joint embeddings. Symmetric endpoint mean and absolute difference
yield 4R edge inputs; a zero-initialized linear projection into the existing
128-wide nonlinear head adds 5120 parameters for R=10. No independent scalar
logit adjustment. No guarantee of correct sign or exact final degrees.

## Frozen one-candidate learning verification

The user's continue/delegated decision authorizes this one bounded local stage.
No reuse of an old expired ledger. Single new 2700-second cumulative window,
including cost probe, learning, generation, evaluation and evidence saving.
Before science, require 1.2*(measured 3000 updates + 48x64 steps + 12 validation
panels)+120 seconds below remaining time minus 30-second reserve.
Only one attempt; bad scientific outcomes do not authorize changing settings.

Same 34-node panel, original 8-layer FP32 model, T64/four blocks, AdamW,
CE/noise/queries and all original optimizer settings. Seeds 2026102011/12/13,
1000 fresh updates each, validation at 0/250/500/1000, final1000 only.
Copy every shared tensor from archived actual capacity-C initial weights,
new zero projection, fresh optimizers. Compare every training input digest
against the archived capacity-C run. Same eight sampling seeds
2026102031..38, two conditions, 48 new graphs; reuse 48 archived capacity-C
graphs as the paired comparator, not new observations.
All query/uniform/RNG schedules checked step-by-step. No extra randomness.

Predeclared decision: report all three trained seeds and both conditions.
Judge actual Genre W1/joint TV, max degree, >=12 hubs/multiple hubs, <=1/zero
degree counts, relation/total edges, closure, components/LCC and topology.
No scalar winner score, quality filtering, checkpoint choice or parameter
search. Lower proxy CE alone is not success. Mixed fidelity/structural
directions mean a tradeoff; no improvement means reject this candidate.
Only three training repeats and seen conditions; no arbitrary target
generalization or large-graph scalability claim.
