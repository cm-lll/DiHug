# Role-condition diagnosis and one candidate correction

Base: `c1163a885322c9dffc2919c8968c6a98adf375c7`.
This directory is independent. `baseline/` is an unchanged copy of its reviewed
role_network_condition_v1_1 package. Original results and weights remain read-only.

## Frozen diagnosis

Use all three existing C final-1000 weights, checked against the published weight
hash ledger. Use all 12 original fixed validation inputs (four held-out graphs,
t=1/32/64), each with its original four-block full plan. For each fixed Xt and
query, run balanced, extreme, balanced again. Save encoded tokens, Q/K/V,
attention, entropy, role context, degree and role residuals, two logits, logit
gap and p0, including separate Movie–Genre summaries. Unlike token sets are not
subtracted rowwise; their weighted first/second moments are compared, and raw
tokens remain available. Attention comparisons are against each target's own
frequency prior. Entropy alone is not informative when token counts differ.

Initial state was not separately saved. Reconstruct original paired_models with
the original seed/environment, verify its whole initial validation mean against
the stored receipt, and explicitly label reconstruction. This is not a claim
to have recovered independently stored initial weight bytes.

Engineering tolerances before candidate tests: integer inputs/maps exact;
FP32 tensor equivariance atol=2e-6, rtol=1e-5; same-input deterministic CPU
repetition should be bitwise equal. Directional mechanism witnesses require
probability differences >1e-5 and correct signs versus an enumerated oracle;
failure is retained, not met by changing tolerance.

## Finding and priority

The measured role contexts differ by 0.0336–0.0486, projected node residuals by
0.00118–0.00566, but final MG p0 differs by only 2.50e-6–1.59e-5. Repeat noise is
zero. Genre attention is very near the prior frequencies; maximum deviations
are below 0.001. Thus aggregation is weakly node-specific but does not erase all
target information. Learned projection and downstream mapping attenuate its
observable effect. This does not isolate a particular normalization layer or
prove capacity/training duration is the cause.

The single prioritized change is **role-context injection at the nonlinear
joint readout**, retaining the old token encoder and attention. Degree features
still enter initial nodes. No additional encoder, loss, optimizer change,
guidance, quota assignment, positional identity or repair is added.

Let c_u be the existing frequency-prior-attention role context (32 dimensions).
For a query uv use the symmetric 64-vector
z_uv=[(c_u+c_v)/2, |c_u-c_v|].
For the existing joint input f_uv and existing hidden width128:

    logits_uv = W2 SiLU(W1 f_uv + b1 + U z_uv) + b2.

U has shape128×64 and is zero-initialized. The old role_out32→256 projection is
removed. Hence parameter count changes by8192−8448=−256. This is nonlinear
joint prediction, not a separate scalar CN/condition logit correction.
CN, degree residual, common joint features, stable CE, support, posterior,
schedule and query plan are unchanged. A fresh matched no-role B control is
needed: zero U must exactly reproduce B with copied common parameters.
Old trained C weights cannot be treated as native weights for this architecture.

Only condition features are query-independent; Transformer query context is
still query-dependent. Context lifetime and target/time/parameter guards are
inherited and tested. Role attention is O(sum_k n_k J_k h), not merely O(E).
Additional per-query projection is O(Q×64×128), replacing node projection
O(N×32×256). No network-hidden caching across changing Xt.

## Directional checks and limitations

A hand-set readout on an exactly enumerable bipartite toy tests whether the
interface CAN express both positive and negative condition-dependent edge
responses using current noisy degrees and anonymous targets only. It is a
capacity witness, not learned coordination. We retain the symmetric-Xt
counterexample: equal edge marginals may be correct despite differing joint
graph laws. A zero final projection must have no response; a learned network
could also ignore U. Late fusion alone does not allocate quotas, ensure one
hub, recover a target distribution, or ensure better complete generation.

The frozen-weight diagnosis made no updates. Tests include finite toy engineering
updates. The user's subsequent authorization adds a <=570s LOCAL one-GPU
full-IMDB resource gate, with five temporary engineering updates per architecture
and one complete reverse step each. These are not trained scientific weights or
complete generation chains. No BW/notebook/private weights are accessed.
Subsequent paired learning panel needs separate budget authorization after review.

## Measured follow-up

All 60 inherited tests and 15 targeted tests pass on CPU, zero skips (20.43s).
The new tests include naturally opening upstream role-encoder/Q/K gradients
after three finite toy updates from zero U; initialization and CE are unchanged.
Nine further fixed-input layer traces (3 seeds x t=1/32/64, original block) show
attenuation across the stack. At seed2026102012/t1 the maximum node difference
is0.00671677 after layer1 and0.000039339 after layer8. Edge representations
need not decay monotonically, so this is not a proof that one particular layer
or normalization causes the failure.

The capacity witness has high-observed-degree edge p0 response +0.00445849 and
low-observed-degree response -0.00445858, agreeing in sign with an independently
enumerated posterior for those evidence classes. Other edges are not claimed
correct; these are hand-set engineering parameters. With completely uninformative
Xt and equal edge totals, the two anonymous distributions have identical edge
marginals1/4 despite different graph laws. The new interface cannot guarantee
unique-hub allocation or suppress multiple hubs.

Original/new parameter counts are6,660,625/6,660,369. On the full8layer model and
original34node state, CPU forward/backward medians (no optimizer step) were
0.16378/0.16272s; complete4block prediction0.15712/0.15652s, three steady repeats.
Differences this small are not a speedup claim.

Full IMDB resource gate used one idle local RTX2080Ti,11264MiB. No visible-device
restriction was relaxed. Full data SHA:
10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42.
N1653/E4079, FP32 original dimensions,T1000/R20, no clipping/AMP/scheduler.
Same normal input construction, noise states t1/500/1000 and temporary fresh
weights with common parameters explicitly copied. Both arms had two warmups,
three measured updates, then one full frozen t500 reverse step.

| Quantity | Original C | Late C |
|---|---:|---:|
| Steady update median (s) |3.03107|3.04892|
| Steady update P90 (s), n=3 |3.03397|3.05271|
| Complete20block step (s), n=1 |4.22651|4.25130|
| Peak allocated MiB |8453.49|8471.44|
| Peak reserved MiB |8670|8736|
| Maximum observed total device usage MiB |9519|9585|

Total parent72.41s; worker48.23s wall/48.10CPU; peak process RSS3582.58MiB.
Total device usage is sampled after synchronized operations, NOT continuous
peak usage. Sampling allocated peaks ~826MiB but reserved memory remains from
training. Device returned to0MiB at exit. Both20block steps covered997313active
slots exactly once; deterministic-none slots retain original support.
No OOM or path-limit exception occurred. The tested states have about4k real
edges; this does not certify a dense generated chain. There is no cross-backend
checkpoint migration here: the existing small weights were CPU-loaded in their
original environment; GPU probes are fresh disposable weights, not BW resumes.

## Optimization vs generation (archived, not rerun)

IMDB fixed5time CE at9k/10k/12k is.00846244/.00953064/.00930835.
Low-t positive CE goes.00473388 at3k to.31599126 at12k, with Movie–Genre
positive CE.56193698 at12k. Thus there is observed fixed-validation deterioration,
not just a bad complete chain. PubMed mean CE at9k/10k/12k is
.02531267/.02522088/.02533532, closer to a plateau; low-t positive CE at12k
is.00850709. The two datasets need not share a single optimization diagnosis.

At12k, low-t validation predicted edges are3910.76/12552.90 for
IMDB/PubMed (reference4079/12284), whereas complete generated graphs contain
2727/9200. Different inputs and a whole stochastic chain intervene. We have no
matched same-time generating-state prediction trace to isolate the responsible
step, so this supports investigating generation-state shift, not claiming it
proved. One graph/checkpoint is not a variability estimate.

A future LR-decay-only comparison can test late optimization drift; EMA-only
can test parameter averaging sensitivity. Neither cause is proven; neither is
included in this architecture correction or its first comparison.

## Proposed next budget (NOT executed)

Prioritize actual **local** generation on the variable-condition toy panel:
original C versus late C,3 paired original training seeds,1000 updates each,
same8layer FP32/T64/R4/data/train-input schedules; common parameters actually
copied. Keep zero/250/500/1000 validation and final8paired sampling seeds for
both conditions per training seed. This is6000updates and96complete small chains.
Retain historical A/B as auxiliary references, not extra independent repetitions.
Use actual graphs for GenreW1, joint role frequencies, maximum degree, Gdegree>=12
count (especially >1 hub), edge counts, LCC/isolates, closure, and type-preserving
topology/duplicate summaries; summarize by training seed. No best-checkpoint
selection, output filtering, or loss-only success gate.

Original CPU panel measured0.23358s/Cupdate and10.408s/fullchain. New CPU
forward/backward and4block costs are essentially unchanged. Forecast roughly
41minutes core compute plus validation/IO/audit; propose one60minute cumulative
local CPU window after review/budget confirmation. This is a forecast, not a
newly started timer or authorization to run now. No small-graph GPU speed is
claimed; current GPU measurement uses full IMDB.

For planning only, full-IMDB12k updates would cost ~10.2 GPU-hours/model at the
measured speed, and1000times the one-step cost ~71min/graph. Two models plus
four paired final sampling seeds (~8graphs) would therefore require ~30GPU-hours
before validation/IO/margin, with dense-state time/memory and single-reference
constant-condition limitations. Do NOT start this large comparison before the
variable-condition mechanism passes actual generation checks and a separate
budget is agreed.
