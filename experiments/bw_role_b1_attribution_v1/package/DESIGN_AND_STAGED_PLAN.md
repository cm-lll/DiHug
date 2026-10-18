# Joint local/global readout v1: implementation and unissued validation stages

This supersedes mean+tanh as a primary fix. Mean is input scale handling;
the actual change is one nonlinear two-logit decision using ALL input channels.
No old CN2 weight conversion, native recovery bypass, automatic training or
scientific full-chain command is provided. Old CN2/source/outputs stay intact.

## Evidence and scope

The notebook path diagnosis_cn_20261016T1508 is inaccessible on this host.
User-reported final336186/reference4079 edges, late-chain6116/26218/71581,
time970 expected net changes and >99% learned-phi contribution are mechanism
evidence on inspected states, not an independently loaded whole-chain causal
ablation. Removing a trained branch is not an independently trained baseline.
No final336186-edge graph is loaded or its~246million wedges constructed here.

## Actual data flow

1. Freeze complete Xt as canonical physical typed edges. Derive existing-edge
   directed graph, exact wedges (only if CN enabled), and complete-Xt statistics.
   Existing-edge/query union remains the pinned SparseDiff tensor coalesce path.
2. Run the original eight Transformer layers on each query-conditioned sparse
   computational graph. Core message/attention/FF formulas are unchanged.
3. Expose normalized final node hidden256 and edge hidden64, plus actual global
   hidden128. Average the two directed edge hidden vectors. Endpoint features
   are mean(h_u,h_v)256 and abs(h_u-h_v)256, hence orientation symmetric.
4. The existing global y starts as a learned projection of t/T. In the eight-layer
   setting the first seven layers update it using learned time/state, node and
   edge mean/min/max/variance pooling. The last layer passes it through. This is
   genuine learned global information over the query-conditioned computational
   graph, NOT a query-independent exact full-Xt connection budget. Its gradients
   reach the common joint readout. Query-block dependence is inherited and stated.
5. Add public complete-Xt statistics for each family r: density m_r/S_r and
   log1p(m_r)/log1p(S_r), with zero for S_r=0. m_r counts physical edges once;
   S_r is type-inventory slot capacity. These 20 values are computed ONCE per
   frozen state, before query assembly. They contain no target degree constraint.
6. CN enabled: use first shared Transformer layer on real Xt edges ONLY for
   contextual h_w256, as in CN2. Each exact common neighbor contributes
   phi([h_w,type_w,onehot(r_uw)+onehot(r_wv),candidate_relation,t/T]) ->32.
   phi is Linear281->32, SiLU, Linear32->32, SiLU on IMDB. Relations are unordered
   around the middle node, preserving endpoint symmetry. Aggregate CONTENT by
   arithmetic mean; add log1p counts by middle type4 and existence1 ->37.
   No CN: no wedge extraction/contextual pass/phi execution, feature37 is zero.
7. Concatenate edge64, endpoint256+256, global128, time1, Xt statistics20,
   candidate-family onehot10, current-state onehot2, previous backbone binary
   logits2 as input information, CN37 =776. ONE Linear776->128, SiLU,
   Linear128->2 predicts both logits. No old-logit residual or CN scalar output.
8. Same binary softmax and exact prior0/1 support, same CE(sum)*query_weight,
   same FP32 network/FP64 kernel and Bayes posterior. T1000/R20, original plans,
   RNG and complete-step commit unchanged. Train/eval/sample all use this code.

Backward-compatible optional vendor return_hidden exposes tensors without
changing the original default backbone result. New model forward/legacy alias
both use the joint head; they are NOT two independent implementations.
Existing backbone node output still has no node supervision. Existing edge
output is retained as an INPUT, so its gradients are meaningful. No feature
from clean labels, reference edge identity or node-ID embedding enters forward.

## Parameters, scale and work limits

Shared head adds99714 parameters to the old baseline6527311 ->6627025.
Optional phi adds10080 ->6637105, versus old CN2=6537427. A/B initialize common
parameters identically under the same seed; optional CN is initialized after
common head. A is a MATCHED NEW FRAMEWORK no-CN control, not unchanged v9.

Mean avoids an explicit linear sum multiplier; log-count preserves quantity.
No tanh cap or hard probability/edge budget is added. Mean does not guarantee
stable predictions. The nonlinear head can learn positive or negative CN
effects and interactions; it can still learn poorly or extrapolate badly.

Per query block, head needs~776*128 multiply-adds/candidate; at~50k queries,
~5.0billion multiply-adds, an engineering cost risk to measure on BW. Both A/B
pay it. Optional CN retains the Xt-only shared first-layer pass and exact wedge
index/phi workload. No query batching or backbone hyperparameter changes.
No all-node-pair learned head or NxNxN allocation is used. CPU slot plans remain
O(S); no claim of end-to-end linear-space arbitrary-large-graph support.

Exact-work guards: <=2000000 Xt wedges and <=2000000 matched paths per block.
Check before corresponding repeat expansion; exceed -> stop, not truncation.
Phi chunks65536 paths are memory batching, every path remains included. Guards
do not guarantee a GPU memory footprint; real peak must be measured. Context
invalidates on state token, close, device/parameter/time changes. Query-specific
hidden encodings are never reused across blocks; Xt-only CN encoding can be
reused within a frozen no-grad sampling step. Training owns one query context.

## Four arms and role version

A: shared new framework, CNoff/roleoff. B: same framework, CNon/roleoff.
C: same A weights, roleon. D: same B weights, roleon. Only two fresh fits.
All train on the same complete IMDB, seed/config/optimizer/loss/precision and
10000 effective updates if separately authorized. Current CN2 and oldv9 are
historical references, not matched trained A/B for this architecture.

The copied role_degree module is byte-identical to role_degree_design_v1:
K32/seed2026101603, known per-type full joint role target, t1-only complete p0
collection, delta.1/max8 trials. No training conditioning or auxiliary loss.
The small adapter test verifies A/C and B/D reuse; device cost checks do NOT
apply guidance. Role guidance must not hide earlier edge inflation. It is a
new terminal law, not original-law/exact target-conditioning. No fixed per-node
role assignment. Actual target distribution and current Xt statistics differ.

## Staged validation: proposals, not automatic authorizations

Stage1 resource check: user confirms one BW process, <=480s worker +3s cleanup,
OMP/BLAS1, unchanged manufacturer Torch/PyG. Fresh engineering random models,
one full-size Xt from fixed forward noise t30 (seed in preflight), same query
plan and input A/B. One update/arm, diagnostics t1/30/1000, one frozen20-block
engineering reverse step/arm. No saved scientific graph. Report initialization,
forward/backward/update/graph/network/posterior timings, features/logits/p
quantiles/saturation, raw/matched wedges, allocated/reserved/RSS peaks. Any OOM,
unsupported operator, nonfinite value, guard violation or timeout -> stop/retain
partial evidence; no auto shrink, precision change, truncation or retry.

Stage2 after explicit approval: fresh matched A/B, fixed256 effective updates
each, original AdamW lr2e-4/wd1e-12/eps1e-8/AMSGradFalse, batch1/noaccum. Same
data/config/seed and paired training input stream. Before any generated output,
freeze independent validation t1/10/30/100/500/1000 with full20-block coverage,
initial/final and fixed checkpoints, relation/positive-negative CE with counts,
constant-prior CE, expected counts/degree tails and CN buckets0/1/2-3/4-15/>=16.
Report actual Bayes birth/death/net counts by relation/time/bucket. Feature
masking within B is a labeled counterfactual, NOT independent baseline or an
additive decomposition. Compare native A/B separately. Only numerical/contract
gates certify engineering, never quality/convergence. Estimate actual cost from
Stage1 before authorizing this budget. No automatic jump to10000 updates.

Stage3 separately authorized: evaluate complete T1000 B/A chain stability using
native fixed weights, report every complete step physical/family edge counts,
component/LCC and late-time probability saturation. Costs/wedges guard checked
before expand. If guard stops, mark incomplete; never truncate neighbors or
mislabel intermediate output as final. Retain all failures. First judge scale,
joint/family degree distributions and tails, connectivity and D-M-A closure
together; then decide whether a formal10000-update four-arm comparison is worth
issuing. Loss/closure alone cannot establish success. Any known-target role
control remains secondary to unguided chain stability and is not a rescue rule.

No dense-stress generated graph is needed for this implementation gate. Optional
future pressure fixtures must be explicitly synthetic, N<=128, <=100000 wedges,
one fixed query/time, <=60s; no extrapolation to historical trajectory evidence.
