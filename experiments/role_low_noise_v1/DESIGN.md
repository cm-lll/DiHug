# Low-noise opportunity test, v1 (frozen before learning)

CORE, single local small-graph stage authorized by “继续推进看看”.
Baseline: capacity-C at b8e97cc; infrastructure parent 4a186b1. Do not
include the unsuccessful degree-input addition. Model and parameter count are
identical: 6,658,481. No new neural module, repair, auxiliary loss or scheduler.

## Question and sole intervention

Previously reconstructed training inputs rarely included low-t peripheral
degree errors seen in generated chains. That is evidence of limited finite
coverage, not proof of the failure cause or of zero mathematical support.
Test whether allocating more updates to low t helps, without changing the
population CE objective. This does not directly add rare Xt at a fixed t.

Original L = (1/64) sum_t E_graph,Xt,query[CE_sum * R/S].
New proposal rho = .5 Uniform{1..64} + .5 Uniform{1..8}.
Thus rho(t)=9/128 for t<=8 and 1/128 otherwise.
Use w(t)=(1/64)/rho(t): 2/9 or 2.
Loss = w(t) * CE_sum * R/S; no renormalization by the current batch.
E_rho[w(t) grad L_t] equals the old uniform-time gradient wherever finite.
This equality concerns the raw stochastic gradient, not identical AdamW
updates/trajectories. Gradient variance and adaptive optimizer moments change.
No promise that variance decreases; high-noise updates receive weight2.

Sampling of Xt conditional on t, forward schedule, support, CE, query selection,
Bayes posterior and complete-chain sampling remain unchanged.
Time proposal is fixed by prior diagnostic range t<=8, not tuned to new results.
This is not a new learned noise schedule and not off-policy teacher correction.

## Pairing, protocol, stopping

Use saved actual capacity-C initial state dictionaries, exact all-tensor copies,
new empty AdamW state. Three original train seeds2026102011/12/13,1000 updates
each;34 nodes,8 layers,FP32,T64,4 queries. Original config/optimizer immutable.
Reuse the48 archived baseline graphs and updates. Reconstruct every original
training input and match its digest; original RNG drives topology/augmentation.
Candidate gets same clean augmented graph, labels and query, but a new time
draw from SeedSequence[seed,update_index,2026102091]. Clone baseline RNG after
its time draw for sparse forward noise. Variable Binomial/choice counts mean
we do not claim equal noise masks or equal Xt. All48 sampling chains retain
old initial state/query/uniform/RNG contracts, seeds2026102031..38.

Validate0/250/500/1000 on the original fixed panel; only final1000 weights
generate. Unconditional loss reported separately from importance-weighted loss.
Primary outcomes: actual Genre W1 and low-degree<=1 count, separately balanced
and extreme. Also jointTV, MG edge count, high>=12 count/multiple hubs,
overall edge error, closure, connectivity, topology classes; all seeds reported.
Three training repeats, not48 independent fits. No significance/generality claim.
No default promotion if extreme W1/periphery fail or improvements trade off with
balanced/structure. No retuning, second candidate, extra seeds or checkpoint choice.

New single2700s execution window includes cost gate,3000 updates,48 chains,
evaluation/audit/save. Cost gate:1.2*(measured full-size costs)+120sec must fit
remaining minus30sec. Five discarded probe updates. STOP/signals/timeouts save
partial evidence; no reset. Tests and code preparation separately recorded.
CPU1thread; no GPU/BW/notebook/large-graph execution. Existing budgets stay closed.

## Numerical checks and limitations

Finite enumeration of128 proposal atoms must reproduce uniform-time objective
and gradients within float64 2e-12. Model gradient scaling tolerance
atol2e-6/rtol1e-5, frozen before execution; integer/hash contracts exact.
Sampler/math source unchanged, all original regression tests retained.
Completed panel measures this proposal only: failure cannot rule out other
low-noise coverage interventions or diagnose the unique root cause.
