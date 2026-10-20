# Frozen-weight wrong-edge recovery diagnostic v1

Baseline: b8e97cc42e6e835e0ea228a6666e3a22a200e035, capacity_C final1000,
three seeds 2026102011/12/13. No training, sampling, degree residual, importance
time proposal, output control, hard assignment or repair.

## Frozen panel, before inference

Use the four archived validation topologies (two per condition), not new graphs.
At t=1,4,8, original T64 forward kernel, one noise draw per graph/time using
SeedSequence [2026102101, graph_index, t]. Original anonymous condition stays
unchanged. One selected Genre node per graph, chosen with seed
[2026102102, graph_index] from clean degree1 nodes (extreme) or all G (balanced).
The selected node is fixed across times. Clean identities are diagnostic labels,
never additional model inputs.

Extreme strengths: add one false MG edge; add enough false MG edges to make the
selected node's current MG degree exactly5 or6. If already at/above the requested
degree or insufficient legal movable absent slots, mark infeasible, no replacement.
Balanced strengths: delete1/2/3 clean true MG edges that still exist at the
selected node. Insufficient availability is infeasible.

For each concentrated edit set, a same-count control edits only OTHER Genre
nodes, in seeded round-robin order, spreading edits across available nodes.
Extreme controls also use clean degree1 nodes: the true degree18 hub has no
clean-absent MG slots. This tests concentration/location, NOT peripheral-vs-hub
specificity. Balanced controls spread deletions across other G nodes.
No edits touch fixed priors or other relation families.

Ordinary plus six variants gives at most84 states ×3 frozen models=252 complete
probability sweeps. All four query blocks in a graph/time triple share plan seed
[2026102103, graph_index, t], identical across variants and models. Complete Xt
is frozen; use production StepPredictor and original kernel.reverse. No uniforms,
state sampling or reverse_step is called. 12 original validation states/model
are separately reproduced at the historical t=1,32,64.

## Gates and cost

CPU single thread; original numerical environment and baseline source hashes
must agree. Check frozen weight receipts from the fixed Git baseline, not hashes
invented from current files. Historical validation rows/counts/CE/expected edges
must match atol=1e-6, rtol=1e-5 (counts exact). Historical full p0 arrays are not
available in these validation JSONs: this gate does not claim their bitwise replay.
Repeat one identical ordinary sweep/model; p0 and q atol=1e-7, rtol=0.
t=1 q=p0 atol=1e-12. All probability/support/coverage checks mandatory.

Single diagnostic invocation limit600seconds, five-second exit reserve, no reset
or overwrite. First original validation/model1 also measures cost; conservative
forecast 2×(remaining validation + 255 sweeps at validation seconds/12) +30sec
must fit remaining time. No model/seed/strength selection based on outcomes.
Engineering tests, packaging and publication are recorded separately.

## Analysis fixed before outcomes

Save every Xt, clean diagnostic label, edit mask, query ids/families/endpoints,
p0 and q. Use identical edit/category masks on each paired ordinary input.
Report per seed first, then equal-weight three-seed means; edge/node observations
are not independent training repeats. Empty categories are NA with count0.

For p0 and q report added false edges, remaining clean true edges, deleted clean
true edges, untouched clean nonedges, plus MG-specific categories. For q, births
=sum((1-Xt)q), deaths=sum(Xt(1-q)), net=births-deaths, globally/per relation and
incident to the selected node and edited endpoints. Verify net=sum(q)-sum(Xt).
The ordinary comparator may have different edge presence, so probability change
and net birth/death change are separate quantities.

Report Genre sorted expected-MG-degree W1 to target role multiset, and complete
joint expected-role vector empirical TV (usually1; descriptive and sensitive to
continuous-vs-integer support). These are statistics of expected degrees, not the
degree distribution of sampled graphs, nor diversity measures. Also report
expected degree per node and target multiset without fixed role assignment.

No binary quality gate or post-hoc threshold. False-edge retention up / true
edge recovery down relative to ordinary are directional degradation evidence;
measure collateral effects and corrected fraction separately. Pressure failure
establishes only a robustness gap for this synthetic distribution, not a causal
explanation of free-chain overconnection or a unique correct topology.
No training follows automatically. Mixed outcomes remain mixed.
