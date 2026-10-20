# Anonymous joint-role network conditioning v1.1 — review before execution

2026-10-20 update: user authorizes a single cumulative maximum of5400 seconds
for the complete LOCAL small-graph learning/sampling/evaluation panel, AFTER
code review passes. This delivery performs regression checks only; the panel
has NOT started and the budget clock has NOT been activated. All scientific
settings are unchanged from v1, verified by an exact configuration comparison
excluding only version and wall-clock limit. Previous30-minute gate receipts
remain historical evidence, not the new budget or an executed90-minute panel.

Additional checks: natural zero-initialized projections remain unchanged;
five ordinary CE/AdamW updates show upstream role/query/key/value gradients
zero on update1 and finite/nonzero thereafter. Query/key gradients are about
1e-10 in this fixture: connectivity is verified, effective learning is not.
Frequency-only changes on identical role tokens and mean degree4 produce a
nonzero probability response with a nonzero test projection. Duplicate-role
merging and switch-back outputs are exact; stale-target contexts are rejected.

The public launcher writes an exclusive, persistent sibling .budget.json
before creating the worker. It records the original activation and cutoff;
an existing receipt is rejected unchanged, so restart cannot grant a new
window. This prototype does NOT implement panel resume. No auto-retry or
fresh restart is allowed after partial completion. All completed logs and
per-seed artifacts remain; partial output is never called completed.
The cap reserves120 seconds plus5 seconds cleanup inside5400, allowing at
most5275 seconds for worker startup/probe/learning/sampling/evaluation/IO.
No new run has been activated by this delivery.

Base: 686f37effdf71b19d9aadf8a1f5e083df1d26394, joint_cn_dual_rounds_v1_1.
This is a network-input experiment, not guidance, degree matching, decoding or repair.

## Review and minimal corrections

1. Use full frozen Xt physical edges for relation degrees, counting both endpoints once. Query-only absent edges never contribute; existing queried edges still contribute.
2. Use schema neighbor capacities for normalization in ALL arms. No target RMS, target mean, clean per-node degree or clean edge identity enters A/B preprocessing.
3. Preserve the anonymous JOINT vectors and integer multiplicities; canonically merge equal vectors. Types, schema and vector/count pairs are the entire target payload. Validate inventory, capacities, endpoint-stub totals and prior0/1 support. These checks are necessary, not a general graphicality certificate for arbitrary new targets; the panel targets are independently realized by saved graphs.
4. Nonlinear role tokens followed by same-type attention preserve more information than mean-degree pooling. Frequencies are inputs and attention priors; they do NOT allocate quotas. Multiple nodes may still choose similar representations.
5. Final residual projections start at exactly zero. Initial predictions therefore equal baseline; inner branch weights initially receive zero gradients through that projection. Nonzero-projection gradient/response tests separately verify actual connectivity.
6. Copy and check common state tensors, not just seeds. Copy B's entire degree branch to C too. No added branch consumes training/sampling RNG.

## Actual data flow and equations

For node i, degree vector d_i(Xt) has R entries. For type k and relation r, capacity c_kr is the other type's inventory, minus one for same-type no-self-loop relations; nonincident dimensions have capacity zero. Encode d/c and log(1+d)/log(1+max(c,1)), with safe denominator max(c,1). Add coarse type one-hot and normalized time t/T to form x_i.

B: degree residual = W_d MLP_d(x_i), hidden width32, W_d maps32 to256 and starts zero.
C additionally encodes every full anonymous role v with the SAME capacity transforms, frequency f=count/n_k and log(1+count)/log(1+n_k). Two-layer nonlinear encoder produces e_v in R32. Attention is softmax(Q(x_i) K(e_v)^T/sqrt(32) + log f), restricted to node i's type. Context is the weighted sum of V(e_v). The zero-initialized32-to256 role projection adds the role residual. No roles are assigned to node identities.

Residuals enter immediately after backbone.lin_in_X and before the first Transformer layer. The same residual also enters the original query-free CN contextual-node pass after its lin_in_X. Original joint edge head, CN aggregation, stable logits CE, loss R/S, posterior and sampling remain unchanged. A passes no residual; B/C share all common parameters exactly at initialization.

No optional noisy-role expected-degree feature is used in this minimal version. Capacity and exact prior support are still checked. Target-role inputs are allowed training statistics: this is known-target conditional generation, with more information in C than A/B. The toy panel varies conditions; this does not solve constant-conditioning identifiability in single-reference training or prove new-condition generalization.

## Chunk and cache contract

One FrozenTypedContext owns one full Xt, time, model parameter versions, mode, device and target fingerprint. Compute node residual once and share over query blocks. Never cache query-dependent hidden states. Each new Xt gets a fresh context; close releases residuals. Target/time/mode changes within a live context are rejected. Parameter updates invalidate it through original version guards. Training constructs the residual with gradients from the complete Xt but supervises only the original sampled query block. Validation/production use the same feature code. This makes the NEW branch query-independent, not the original full backbone.

Degree statistics cost O(E+NR). Degree MLP costs O(N(2R+K)h+NhH). Role encoders cost O(sum_k K_k R h); attention costs O(sum_k n_k K_k h), and output O(NhH), with h32/H256. Thus the whole branch is NOT O(E). No role truncation or path truncation is introduced.

## Frozen panel and analysis

34 nodes: D4/A6/M18/G6. Ten unordered type-pair families, original active-from-reference support. Fixed DMA skeleton has DA10, DM18, AM36. MG24, Movie MG degrees six2/twelve1, Genre balanced six4 or extreme18/2/1/1/1/1. All graphs have88 physical edges. MG has no common-neighbor signal under this support. Six exact type/relation-nonisomorphic topologies per condition are created by deterministic legal bipartite switches preserving margins; first4 train, last2 validation. Relabel augmentation is AFTER this split. Saved panel contains every edge, ID and anonymous target; exact isomorphism verifies split separation.

Frozen full plan: 3 paired training seeds, 1000 updates each A/B/C, validation0/250/500/1000, T64/four blocks, final8 paired sampling seeds/condition. A/B outputs are reused across target evaluations:96 actual chains,144 evaluation rows, not144 independent graphs. Seeds/config are fully recorded in package/role_condition/config.json. G>=12 is the predeclared high-degree threshold. Evaluation uses actual binary graphs: per-family edges, typed relation W1 and joint TV, tails, components/LCC/isolates, typed triangles, DMA path and unique-pair closure, exact short paths. Different invariant signatures only prove a lower bound on topology counts; equal signatures do not establish isomorphism. Training-seed-level summaries precede paired comparisons. No new quality threshold/composite score is added.

## Cost gate outcome and evidence boundary

Full original width/depth was preserved. Disposable engineering probe:8 updates/arm,8 forwards/arm,8 complete four-block dense-support reverse steps/arm, one fixed validation panel/arm. This is24 engineering updates, not the requested scientific learning panel or complete chains. Forecast includes training9000, sampling96*64 complete steps,36 validations, safety1.25 and20s overhead. Gate failed before scientific training. No shrink, seed deletion, quality-driven stopping or extension occurred.

The delivered input branch and engineering invariances are tested. No trained condition-control effectiveness, actual generated tail improvement, convergence, cross-condition generalization or topology-diversity improvement is claimed. Original T1000/20 compatibility is one bounded reverse step per arm, not a complete chain. No GPU/BW, IMDB/PubMed runs, private weights, guidance or repairs were used.

## Recommended next decision (not executed)

1. Decide whether to authorize approximately80 minutes of single-thread CPU time for the frozen plan, or a separately approved device cost probe. Main cost is full unchanged backbone, not the~1.55ms role attention.
2. Run that frozen paired plan before changing design: discriminate actual tail response, role averaging and ordinary-condition/closure tradeoffs from interface capability.
3. Only then consider one change at a time. Current evidence cannot diagnose ignored conditioning or insufficient training because the learning panel did not run.
