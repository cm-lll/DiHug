# Authorized 6000-update paired panel: 2026-10-21 amendment

This section supersedes all update counts, budgets and release stop instructions
in the historical proposal below. The model, perturbation, loss, optimizer, LR,
uniform time distribution and sampler are unchanged. User authorized execution.

User additionally authorized all eight local GPUs. All eight2080Ti cards were
idle, but the bounded GPU parity gate exceeded the PREDECLARED gradient tolerance:
max2.519169e-6 versus2e-6 (five elements), while logits/p0 passed. No tolerance was
relaxed, and no GPU scientific training is authorized by this release's gate.
Use the validated CPU numerical path, three independent single-thread seed
workers in parallel. Each worker keeps both arms and their update transaction
together. This changes scheduling only, not random inputs, optimizer ordering
within a model, or scientific scope. One global eight-hour ledger is inherited
by all workers; no per-worker clock reset. Global outputs are merged with explicit
path remapping and all288 graphs audited jointly. No extra seed or experiment.

3 original training seeds ×2 arms ×6000 optimizer updates=36000 updates and72000
training query forwards. Retain initial weights. Atomic complete paired
checkpoints at1000/2000/3000/4000/5000/6000 include both models, both continuously
running optimizer states, all RNG/private data RNG, individual counts, source/
config/data/environment binding and the original budget activation/deadline.
Validation at0/250/500 and every500 thereafter. No optimizer reload during training.

After each seed finishes uninterrupted6000-update training, evaluate read-only
model copies from1000/3000/6000. Each point produces96 chains across all seeds,
288 total. Same sampling seeds/query plans/per-slot uniforms across arms AND
checkpoints. Snapshot/restore all training RNG around evaluation; no selection,
repair, extra sample, or result-dependent budget/setting change.

One cumulative28800-second window starts at formal panel parent activation,
including import/probe/training/validation/all evaluation/audit/saving. No old
learning panel was active; frozen/probe runs were engineering only. The full
new cost gate includes36000 updates,288chains,84 validations, final heldout
pressure panel, measured complete checkpoint serialization/reload cost, audit
allowance and15% safety factor. Reject if it does not fit remaining time.

Active STOP and catchable termination are deferred until BOTH arms complete
the current two-microbatch optimizer updates. Hard deadlines still interrupt;
save individual counts and partial state without claiming a complete pair.
Parent waits for graceful boundary on STOP; it still kills its own process
group at the hard deadline plus at most5seconds cleanup. No automatic resume,
budget reset, seed replacement, extra candidate, BW/notebook/large-graph task.

Results must compare normal2 training extension and robust-vs-normal2 at all
three fixed points.6000 is a budget endpoint, not convergence certification.
Complete local results, archive/ZIP/hash and independent GitHub review commit,
then stop. Partial/failed outputs are retained and explicitly incomplete.

---

# Historical proposal v1 (1000 updates /90minutes; no longer authoritative)

Status: preparation and bounded engineering tests only. The 90-minute learning
panel is NOT activated or authorized by this release. No scientific training or
complete generation chain is run during preparation.

## Hypothesis and scope

The b8e97cc capacity-coupled soft-readout model tends to preserve current edge
presence in the 72c7648 frozen diagnostic. That diagnostic is not a causal
explanation of free-chain overconnection. Test whether source-traceable corrupted
inputs improve actual free generation without sacrificing hubs/structure.

Keep the complete model, CE, optimizer, T64/R4 schedule, uniform time sampling,
precision, and ordinary unmodified sampler. No degree residual, importance-time
proposal, extra structural loss, quotas, matching, controller, or postprocessing.
Both models receive the original anonymous condition; this is known-condition
generation on two seen conditions, not arbitrary-condition generalization.

## Scientific comparison

Two arms, same 3 archived random initializations, independent fresh AdamW states:

* normal2: two independent ordinary-noise inputs, average original slot CE.
* robust: same first ordinary input; same second input with artificial corruption.

For each update choose the same training topology and type-preserving augmentation
in both arms. Two independent t draws remain Uniform{1,...,64}; query plans,
labels and underlying forward noise are shared between arms. Ordinary 2nd input
is paired with its artificially corrupted version. Each loss is

    L = (1/2) [(R/S) sum_{i in Q0} CE_i + (R/S) sum_{i in Q1} CE_i].

Backward sequentially with factor1/2, one optimizer step. This is identical to
the ordinary mean of two original unbiased query CE estimators. Do not divide
by sum of weights, rebalance labels, oversample edited slots, or add another CE.
The robust arm's second distribution changes, so its objective is NEW. The new
normal2 arm is the primary comparator; old single-input1000 results are history.

Every arm: 1000 optimizer updates, two forwards/backwards per update, effective
microbatch1×accumulation2. Total6000 updates/12000 query forwards. Width/layers,
FP32 network, float64 kernel, AdamW2e-4/wd1e-12/eps1e-8/betas(.9,.999)/AMSGradFalse,
no clipping/scheduler. Configured dropout.1 is inherited, actual forward dropout
is disabled in the existing adapter. Check no Torch RNG consumption during forward.

## Exact artificial distribution

Only MG slots, for BOTH conditions and ALL sampled times. Direction add/delete
is uniform, independent of condition/time. Strength uniform over add[1,4,5] or
delete[1,2,3]. All settings fixed before cost measurement.

Add candidates: absent in clean graph AND current Xt. Delete candidates: clean
true AND still present in Xt. Require movable prior in(0,1). Choose uniformly
among G nodes with at least the requested number of eligible slots, then choose
that many incident slots uniformly without replacement. This conditional node
selection is part of the distribution; it need not yield identical role-position
mixes across conditions. Log them, do not infer equal role exposure from equal
direction probabilities. If no node is feasible, use unchanged input, count it
as infeasible, never redraw/reduce strength. No per-node role labels are input.

Training seed streams: [2026102111,train_seed,update,stream0or1] for original
noise; [2026102110,train_seed,update] for artificial edits. Clean graph identifiers
and edge labels are used only to construct/evaluate supervised corruption.
Model receives only its usual Xt/types/time/anonymous condition. Edited-slot masks
are diagnostic only. Every normal query CE includes edited and untouched positions;
small queries may miss all edits. Log query hit counts, positive/negative CE,
time bins, feasibility, selected current degrees, and actual edit counts.

## Validation and actual generation

Freeze original validation seed/times1,32,64; records at0/250/500/1000 for both
arms. Restore all training RNG after validation. No checkpoint selection: final1000.

Old four diagnostic graphs remain development regressions, not an independent
confirmation set. Deterministically generate 2 additional heldout topologies per
condition with the same DMA skeleton and joint role multiset. Fixed seed2026102112,
at most1000 degree-preserving swap attempts/condition. Exact typed isomorphism
rejects every old train/validation topology and already selected holdout. Failure
stops preparation, no changed graph quota/search seed. These new topologies are
small development holdouts, not unseen-condition or large-graph generalization.

Heldout stress at1,4,8: ordinary/add1/add5/delete1/delete3. Independent fixed
noise/query/edit seeds2113/2114/2115 (full values in JSON); identical inputs across
arms. Fixed whole-query coverage; exact edit infeasibility retained as NA. This
tests both directions under both conditions without training on these topologies
or edit masks. Original fixed validation retains normal denoising evidence.

Generate96 full T64 chains: 3train seeds×2arms×2conditions×8paired sampling seeds.
Original whole-step frozenXt, uniforms/query plans paired; no output selection.
All actual graphs retained, no repair. Evaluate after generation, not expected
degrees: signed and absolute MG edge error, all relation edge errors, Genre W1,
complete joint-role TV, max/low<=1/zero G degree, Gdegree>=12 count/multiple hubs,
CC/LCC/isolates, typed triangles, DMA path and unique-pair closure, MP2/MP3 and
typed isomorphism classes. Repeated-node numbering is not topology diversity.

Aggregate eight graphs within training seed first; report all3 paired seed
differences separately, balanced/extreme separately. No edge/node-level bootstrap
or96-independent-training claim. Success needs reduced systematic excess and
improved discrete-role/tail behavior with preserved hubs and no clear collateral
loss in balanced condition, normal CE, other edges/closure/connectivity/diversity.
No new composite score or post-hoc tolerance. Any material opposed changes are a
tradeoff. If only stress recovery improves, generation benefit remains unproven.

## Cost, stages and stop point

Historical CPU1thread:3000 one-input updates708.685sec,48 chains+evaluation523.686sec,
12 validations31.683sec. Two arms/two inputs forecast2834.739+1047.372+63.365sec
=65.758min before new stress/preflight/audit overhead. Proposed single new learning
window5400sec includes launch/imports/preflight/training/all generation/evaluation/
saving/cleanup. No previous budget is reused. Five-second parent cleanup allowance
is separately reported; worker reserves30sec. No restart or automatic resume.

Preparation probe has separate engineering cap300sec, cannot activate learning.
Use full34-node/8layer weights as disposable copies, 5 paired two-input updates
per arm (20microforwards), and complete4-block probability sweeps (no sampling),
with separate first/warm/stable timing and source/env/hash verification. Do not
call this scientific training or complete-generation throughput.

Before any future learning activation, the complete implementation/config must
be reviewed. Its CLI requires explicit --execute-learning-panel. At activation,
new output directory, freeze source/env/input hashes and start a single5400sec
parent deadline including imports. Rerun cost gate with disposable copies:
1.15×(6000update forecasts +96chain forecasts +24validations +360heldout probability
sweeps + independent final audit forecast)+120sec must fit remaining minus30sec.
Do not shrink groups, seeds, graph, queries, precision or model if gate rejects.

No automatic BW/notebook/GPU access. Any failure retains partial counters,
weights, RNG and receipt, never promotes partial results to completed. This
release's stop point is code, tests, bounded cost probe and independent ZIP
review publication. Actual quality/90-minute panel results remain unmeasured.
