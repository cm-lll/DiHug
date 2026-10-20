# Capacity-coupled role reading: actual control improves, structural tradeoffs remain

Execution source0efd5b4, based on f2f58a39 and its unchanged late-C backbone.
This is one new local development candidate, not a parameter search, an
independent confirmation, or an IMDB/BW result. The user delegated method choice.

## What changed

Replace near-frequency learned query/key role attention with exact soft occupancy
marginals under a capacity-constrained product-of-node-degree-likelihood working
distribution. The individual forward degree laws use the original noise kernel.
Use the resulting role context at the existing nonlinear late joint head.
The working distribution is not the exact joint graph posterior, since degrees
share edges. No hard node-role assignment, permanent ID, edge identity, degree
repair, extra loss or modified reverse probability is introduced.
All unmodified initial parameter tensors are copied from the actual archived
late-C initial files. Removing query/key tensors reduces6660369 to6658481 parameters.
The role encoder/value and readout still learn using the original stable CE.

This construction gives the network a meaningful per-node role context rather
than nearly the same frequency average everywhere. It does not force the final
graph to satisfy the role distribution. Exact symmetric observations still give
symmetric soft contexts; no hidden random or numbered tie breaking.

## Diagnosis before changing the model

Six existing weights,96 fixed reference-forward-noised condition evaluations:
at t1, late C can retain an already existing degree18 hub with expected degree
16.222/17.843/17.657, while role attention deviations from frequency are<.005.
This supports a specialization hypothesis, not a proof of full-chain causality.
The states were not reverse-chain intermediates. Diagnosis27.83s wall,27.78s CPU,
peakRSS701.16MiB, zero updates/chains/GPU.

## Frozen actual-generation comparison

3 original training seeds2026102011/12/13,1000 new updates each. Full34-node
panel, original8layers/FP32/T64/R4/AdamW/CE. Validation0/250/500/1000. Final1000
only.8 paired sampling seeds per condition per model:48 new complete chains,
48 archived late-C comparator chains. Actual training-input digests and every
sampling step's uniform/query/RNG match the archived comparator. Initial tensors
were independently reloaded and compared, not reconstructed from seed alone.

Values are per-training-seed averages of8 graphs. Arrows late C→new capacity C.

|Seed|Condition|Genre W1|Genre joint TV|Genre max-degree mean|
|---|---|---|---|---|
|2026102011|balanced|1.72917→2.10417|.85417→.87500|5.625→4.375|
|2026102012|balanced|1.47917→1.37500|.81250→.72917|6.750→6.500|
|2026102013|balanced|1.27083→1.41667|.75000→.83333|6.250→5.875|
|2026102011|extreme|3.43750→.68750|.70833→.54167|5.750→17.750|
|2026102012|extreme|4.27083→1.52083|.87500→.60417|6.750→18.000|
|2026102013|extreme|4.47917→.87500|.93750→.41667|6.250→18.000|

Extreme W1 improves in all3 training repeats; macro4.06250→1.02778.
24/24 extreme outputs have exactly one Genre node degree>=12, compared with0/24.
Balanced0/24 have any such node. No output has multiple hubs.
All24 balanced/extreme paired outputs change graph (baseline1/24).
These96 compared outputs represent3 paired training repeats, not96 independent
trained models. Conditions are seen during training; no new-target generalization claim.

## Costs in fidelity, not a global success declaration

|Condition|Mean edges target88|Edge MAE|Closure target.44444|Connected/24|LCC|Isolates|
|---|---|---|---|---|---|---|
|balanced|81.5417→75.7500|11.7083→16.4167|.38364→.37097|16→13|33.5833→32.8750|.4167→1.0417|
|extreme|81.6250→83.9167|11.6250→11.0833|.38364→.36247|16→15|33.6250→33.4583|.3750→.5417|

Balanced W1 worsens1.49306→1.63194, driven mainly by seed11.
Extreme low-degree(<=1) Genre count improves.4167→1.5833 but target is4.
Extreme MG edge mean24.625→28.9167 (target24): peripheral nodes remain too dense.
Triangular structure: balanced12.125→11.7917, extreme12.125→11.4583 (reference16).
Other type joint-TV (balanced): Director .88542→.83333; Actor .81250→.85417;
Movie .94213→.93056. No hiding unfavorable types or conditions.
All6 new seed/condition cells have8 distinct true-type/relation-preserving
topologies, frequency[1,1,1,1,1,1,1,1], effective count8. No timeout occurred;
small samples cannot establish global diversity preservation.

Fixed-input MG probability responses now reach.69420/.78864/.82753 across3 seeds;
repeat-forward noise0. These support interface use, but actual graphs above
are the efficacy evidence. Final CE does not improve for every seed:
.203764→.208399, .200926→.195468, .200244→.194267.

## Acceptance, resource use, failures

14 targeted checks pass, including exhaustive capacity marginals, noisy degree
PMF enumeration, joint-versus-marginal counterexample, permutations, exact symmetric
counterexample, zero projection, nonzero response, gradient flow and resource refusal.
60 baseline+15 late-fusion+14 new =89 checks pass, zero skipped,7 warnings.
The first full test invocation missed PYTHONPATH and failed collection; its XML
is retained. Corrected the invocation, not the mathematics.

First cost receipt serialization failed on NumPy bool before any scientific update.
Source fix0efd5b4 casts JSON scalars. Original activation/deadline and directory
retained; no reset. Both engineering attempts used disposable preflight updates,
not counted in3000 science updates. No scientific retry or incomplete chain.

Independent audit of48 raw graphs: schema, endpoints, all degrees/family counts,
components, triangles and DMAclosure exact. All-type joint-TV max error2.22e-16,
all-type relation-W1 error8.88e-16.1446 common initial tensors equal; initial fixed
validation CE difference0. Actual pairing and all6 weight hashes checked.

One new2700s ledger, NOT a reopening of the completed60-minute experiment.
Panel parent elapsed1387.66s, core worker1300.18s wall/1300.13s CPU.
Stable update median.23469s/P90.24081s;3000 update timers708.68s,48 chain/graph
timers523.69s. These sub-timers exclude some setup/validation/IO; worker totals
include them. Main worker peakRSS647.86MiB; failed preflight699.17MiB.
Independent graph audit.90s; additional tensor/joint audit.94s.
No GPU, BW, notebook, large graph, private weight access or new dependency install.
Python3.9.23/Torch2.1.0/PyG2.3.1/NumPy1.26.4/SciPy1.11.0, CPU one thread.

## Decision and boundary

Keep this as the preferred conditional-role mechanism candidate: it changes
actual generated tails consistently across the3 tested training seeds.
Do not call the overall fidelity problem solved. Balanced under-generation,
extreme peripheral overshoot, closure and connectivity costs remain.
The exact capacity DP has a200000-cell resource guard and exponential dependence
on role multiplicities. It is a small-panel reference, NOT a proven large-graph
implementation; no approximation or truncation was smuggled into this test.
No claim of arbitrary-condition generalization or complete role satisfaction.
Stop after publication. Any scalable approximation or further learning is a
separate next decision; no automatic second candidate or big-graph rollout.
