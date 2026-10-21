# One correction: capacity-coupled anonymous role reading

Track CORE, local conditional heterogeneous graph-state diffusion development.
Based on f2f58a39 / late-fusion candidate at1613be1. User delegated the next
decision after the failed paired panel. No notebook/BW/big graph work.

## Evidence and selected hypothesis

Six frozen models,96 reference-noised condition evaluations at t1/8/32/64:
at t1 the late model predicts16.22/17.84/17.66 expected edges for an existing
degree18 Genre hub; attention deviates from frequency by<.005. Old full chains
produced no Genre degree>=12. This supports testing role specialization rather
than another injection location. It does NOT prove exact reverse-chain cause:
these new states are reference-forward-noised, not saved reverse intermediates.

## Model

Replace only learned query/key role attention in the late model. Keep degree
branch, nonlinear role encoder/value, late joint head, CN,8layer FP32 model,
CE, AdamW, T64/R4 and original schedules. No clipping, EMA, new loss, probability
controller, guidance, edge fixing or new sampling RNG. Query/key parameters are
removed; all other actual initial tensors copied from archived late-C initialization.

For type k, anonymous joint role vectors v_j have integer counts n_j.
At current noise time t, retention a_t and prior pi_r, the observation law is

  D_tr | v_jr ~ Bin(v_jr,a_t+(1-a_t)pi_r)
                   + Bin(N_kr-v_jr,(1-a_t)pi_r).

N_kr is legal schema capacity; same-type capacity excludes self. The readout
uses log L_ij = sum_r log P(D_itr=d_itr|v_jr). Only full frozen Xt real edges
provide d; absent query edges are never counted. Target has no IDs/edge identity.

Define a working measure over all assignments z with exact role multiplicities:

  w(z) = exp(sum_i log L_i,z_i) / Z,  count(z=j)=n_j.

Compute A_ij=P_w(z_i=j) exactly by log-space forward/backward capacity dynamic
programming, not by a hard assignment or independent per-relation matching.
Feed sum_j A_ij value(encode(v_j,n_j/n_k)) to the SAME late nonlinear joint head.
Every new Xt recomputes A; no sampled/persistent node-role mapping. Column
occupancies are constrained in the SOFT READOUT, not generated graph degrees.

The single-node degree law is exact for the marginal forward noise. The product
across nodes is a composite likelihood because endpoint degrees share edges;
w is NOT claimed to be the exact latent-role posterior of the graph generator.
No universal role-control or degree-distribution guarantee follows.
Identical likelihood rows remain identical, including exact symmetry. At a=0
the role likelihood does not depend on the role, and A is the role frequency.
This retained counterexample prevents fabricated deterministic symmetry breaking.
No temperature or new fitted coefficient is introduced.

## Numerics, complexity and limits

Float64 log probabilities and logsumexp; binomial probabilities0/1 exact,
no epsilon. Impossible observations fail explicitly. DP eliminates the largest
capacity dimension and uses O(n K product_{j<K}(n_j+1)) work; enforce maximum
200000 forward table cells, never silently truncate roles. This is a small-panel
prototype, NOT yet a scalable arbitrary-IMDB solution. Tables cache only immutable
forward degree laws keyed by capacity, role degree,time,prior. No hidden-state cache.
Model context lifecycle/target checks remain inherited.
Row residual tolerance2e-10, column2e-9, neural comparisons atol2e-6/rtol1e-5,
fixed before execution. Precision/optimizer unchanged.

## Frozen learning comparison, no search

3 original seeds2026102011/12/13;1000 updates each; validation0/250/500/1000.
Use actual archived late-C initial weights and identical training input digest
for every update. Independent empty optimizer. Original forward consumes no Torch
randomness; check this on every update in both new and archived logs.
Use8 original sampling seeds2026102031..38 at both conditions,48 new full T64
chains. Reuse48 archived baseline graphs/trained weights, not rerun baseline.
Check initial prior bytes and every step's query/uniform/RNG against old receipts.
Reuse is legitimate paired development comparison; it is not an independent
confirmation of a hypothesis chosen after seeing the baseline failures.

One new2700-second window for learning, validation,48chains,evaluation,saving
and audit. Old60-minute panel remains closed. Cost gate: full8layer model,
5 disposable updates,3 dense complete reverse steps, full fixed validation.
Forecast1.2*(3000*P90update+48*64*P90step+12*validation)+120 must fit remainder
minus30seconds. Otherwise no scientific training. No retry with a new window,
shrunk scope, picked seed or changed coefficient. No automatic second candidate.

Evaluate actual graphs, by training seed and condition: GenreW1 and full joint
role TV, max degree, degree>=12/hub multiplicity, low<=1, per-family edges,
closure, triangles, connectivity/LCC/isolation, topology duplicates. All48 graphs
retained; final1000 only. Improvement needs actual extreme-tail/role evidence
across seeds, not just a lower loss or stronger probability response. Expose
balanced-condition and ordinary-structure costs; no posthoc composite score.
No longer run or large-graph rollout if this does not work.

## Reproduction

Existing weight diagnosis:
  python diagnose_states.py --run <archived_paired_run> --output <new_diagnosis>
CPU candidate panel (one exclusive new directory, no resume/reset):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python panel.py \
    --source <archived_paired_run> --output <new_candidate_run>
Stop: touch <new_candidate_run>/STOP. Interrupted results remain incomplete.
Do not execute these as an automatic follow-up to the already completed run.
