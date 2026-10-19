# Local frozen A1/B1 interface v1.1

This is a STANDARD local-file contract, tested with explicitly SYNTHETIC fixtures.
It is not a claim that the old real export has been connected or authenticated.
No network, SSH, download, notebook path search, denoiser or chain exists here.
Definition/source baseline: 2c1a2f2c0c2a4d82402979a03f41be0ad6b02586.

## Caller trust root and files

Pass --input MANIFEST (or its explicit parent root), --manifest-sha256 EXPECTED,
--arm A|B, --output NEW_DISJOINT_DIR. EXPECTED must come from an independently
sealed receipt, not from hashing the current untrusted manifest at invocation.
Manifest has files.payload, files.target, files.source_receipt; each is a relative
local path and EXPECTED sha256. No path may escape this explicit input root.
All actual input bytes are checked against these pins and rehashed before success.
No old ledger/manifest is written or repaired. Missing fields are errors.

Manifest keys:

* format: local_frozen_role_decode_v1
* arm: A or B, t: integer1, unguided: true, seed: nonnegative integer
* T:1000, query_rounds:20
* p0_semantics: clean_edge_present_probability
* q_semantics: next_state_edge_present_probability (required if q is present)
* endpoint_convention: unordered_u_lt_v
* relation_ids: zero_based_family_pairs_order
* score_semantics: recorded_probability OR finite_clean_logits
* query_sha256: original complete query digest described below
* semantic_types: explicit distinct director, actor, movie type IDs
* files: the three path/hash descriptors above

source_receipt repeats arm, seed, t, unguided, query_sha256; includes
source_kind=historical_frozen_output (synthetic_fixture only for engineering),
weight_authentication=historical_receipt_only, and weight_sha256,
source_manifest_sha256, config_sha256, reference_sha256, alpha_sha256.
The reader checks actual payload/reference-target receipts, not unavailable
weight bytes: “沿用历史来源记录，未重新认证权重字节”. These fields are not proof
of authenticity if the caller does not possess an independently trusted pin.

target.json: {profile: ANONYMOUS_PROFILE, reference_sha256: PIN}. Profile is the
baseline anonymous-joint-relation-degree-v1 object with family_pairs and groups
of integer type IDs -> nodes, unique COMPLETE R-dimensional rows and integer
counts. No node-role assignment or reference edge IDs is included.

## NPZ payload

allow_pickle=False. All probabilities, prior, alpha and optional logits are FP64.
Arrays are:

* types[N] integer; pairs[R,2] integer, one unordered coarse-type pair per relation
* slots[2,S] integer, each pair u<v; families[S] zero-based integer labels
* state[S] binary X1, p0[S] clean present probability, support[S] integer -1/0/1
* prior[R] reference relation density; alpha[1001] cumulative schedule, alpha[0]=1
* query_block[S] int64, -1 only for inactive structural-zero families
* query_ids[Q], query_families[Q], query_slots[2,Q], query_offsets[21], all int64
* optional q[S]; optional clean_logits[S]

All legal schema slots MUST be present, including fixed-zero relations.
Slots may be stored in any order: the reader explicitly maps endpoints to
schema.slots and checks family labels. It rejects duplicates, missing slots,
self edges, reversed/illegal endpoints and wrong relation IDs. state/p0/q/
support/query_block follow the INPUT slot order. types/pairs are not silently
remapped to another schema or node inventory.

Query IDs, unlike per-slot arrays, are original canonical TypedSlots IDs.
The full20-block recipe is locally checked against the old source implementation:
SeedSequence([seed,1,61]).spawn(R), per-active-family permutation and balanced
quotas (including old RNG consumption), then concatenation in family order.
Active means reference family edge count>0, NOT an independent schema prohibition.
For each block digest.update(ids.tobytes()), then families.tobytes(), then
query[2,Q].tobytes(), exactly old worker ordering, native little-endian int64.
No new framing. Inactive slots stay in the full payload with exact support.
Complete bytes, per-slot blocks, coverage, labels AND digest must agree; merely
20 blocks or the correct length is insufficient. Legitimately reordered payload
slots do not change original query IDs or the query byte stream.

prior must equal the reference density implied by the anonymous target. Exact
prior0/1 fixes its family to0/1; support/state/p0 must agree. Free p0 must be
strictly between0 and1 in recorded_probability mode. A free numerical0/1 is NOT
automatically a hard constraint. It is rejected unless finite_clean_logits is
explicitly selected, the finite clean_logits array is supplied, and the source
receipt states logits_semantics=finite_clean_present_minus_absent. The reader
then retains finite-logit support and verifies sigmoid/p0 within5e-14. This mode
scores the finite-logit Bernoulli law, not a newly declared structural zero/one;
the number of rounded boundary slots and scoring semantics are reported.
No epsilon or support relaxation is used. Saved q, if supplied, must agree with
p0 at t=1 within the OLD absolute5e-14 tolerance; no relative tolerance added.
At t=1 the next state is s=0, so this is the original posterior identity, not
a new posterior implementation. Other time steps are rejected.

## Optional original unguided output

Supply all four: --original-graph, --original-sha256, --original-receipt,
--original-receipt-sha256. Both expected pins must be externally provided.
NPZ contains types[N], edges[2,m] with u<v and no duplicates, families[m].
Receipt binds arm, seed, t=0, unguided=true, graph_stage=raw_final_t0,
input_manifest_sha256, source_receipt_sha256, graph_sha256 and weight_sha256.
Types, schema, labels and model support are checked. Historical receipt binding
is NOT replay verification of the sampling RNG or reauthentication of weights.
Missing original output means no paired edit/quality/score comparison; the
reader does not draw a substitute. C/D, wrong arms/seeds or repaired graphs
without the declared raw-final provenance are rejected.

## Matching and exact discrete diagnostics

mu_ir=sum of ALL clean p0 slots incident to i in family r, both endpoints once.
For type c, expand complete target vectors by integer counts; scale_cr=
1+sqrt(sum_k count_k*d_kr^2/n_c). Effective dimensions are positive target-mass
relations; excluded zero dimensions are exactly structurally0, still retained
in complete vectors. Minimize sum_i,r ((mu_ir-d_assignment(i),r)/scale_cr)^2,
mean over effective dimensions and nodes, then equal mean over types. Hungarian
joint cost for multidimensional roles; sorting only for one effective dimension.
One allocation is shared by ALL relation decoders; no per-coordinate role mixing.
No alternating optimization, new weight or result-driven parameter choice.

Duplicate identical roles are harmless. Stable sorting/SciPy may choose an
index-dependent representative in a different-role tie. Small<=7-node types
enumerate distinct role allocations within1e-12; large multidimensional ties are
explicitly unresolved, NEVER certified equivariant just from equal objective.
This remains fixed-matching MAP, not anonymous-global MAP or random sampling.

All decoder settings remain frozen_config.json. Report exact assigned integer
degrees, whole joint frequencies, schema/support, primal/dual bounds and solver
status. Timeout/infeasible/uncertified-optimal is nonzero, no successful report.
Exact diagnostics: relation edges; all-node, per-type and per-family-endpoint
degree histograms, maxima, zero/<=1 counts, P90 and descriptive P(D>153); complete
joint counts; typed triangles (unordered type triples); components/LCC/isolates.
D-M-A closure denominator=sum_M deg_D(M)*deg_A(M), numerator counts those paths
with their D-A edge present; zero denominator yields NA, never0. Edge edits by
family and both logq scores use the SAME frozen law. None is a new quality gate.

## Known old-format status, static review only

Local source reviewed: legacy.py convert_arrays/original_query, portable.py
Portable.record, INPUT_CONTRACT/DYNAMIC_INPUT_CONTRACT, hsgd/slots.py and
verify_original_q. Their file hashes are in format_reference/source_review.json.
Old RAW baseline.npz has state,p0,q0,types,pairs,prior,alpha; collected.npz adds
coverage and collection.json supplies time/query/source receipts. Old CONVERTED
six_states NPZ has state,p0,q,slots,families,types,prior,alpha,query_block,t.
Both need their actual externally sealed manifests/receipts; filenames are not
authentication. Neither is auto-detected or silently treated as this standard.

Not present as standard fields in those old arrays: explicit support semantics,
free-boundary finite-logit evidence if needed, flat ordered query arrays/offsets,
the explicit unguided present-probability declaration, semantic type map and
optional final graph provenance. Some may be deterministically reconstructed
from original pinned code and collection by the other side, with origins and
old hashes recorded; reconstruction is not evidence they existed originally.
No real A1/B1 arrays or original output were read here. This release provides a
strict standard consumer, NOT a claimed verified raw/converted legacy adapter.

## Runtime and stopping

One explicit arm per invocation, CPU/one thread, <=300 seconds INCLUDING startup,
hashes, matching, solver, exact evaluation, save and cleanup. Work reserves up to
5s for cleanup. Fresh output is rejected before any writes if overlapping source
or input roots. SIGINT/SIGTERM/SIGHUP and OUTPUT/STOP terminate only owned process
groups; SIGKILL/machine loss cannot be caught. Partial graph files may remain as
incomplete evidence, never successful output. No automatic retry, resampling or
budget extension. Success requires a nonempty check set, graph/hash evidence,
exit0 and complete controller receipt. Ordinary MAP says nothing about diversity.
