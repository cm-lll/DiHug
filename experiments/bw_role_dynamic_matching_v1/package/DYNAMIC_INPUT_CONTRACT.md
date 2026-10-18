# Portable offline inputs (real arrays currently NOT available locally)

Three immutable user pins are in `dynamic_role/portable.py`:

- six_states.json: 25bc708337898e84cf22ceffaa8e39ed815adf4fc916b684503c0afa3c35d7dd
- original_evaluation.json: 198ff9bfcba007a04f36662fc66a9399578b6772b6a5c19be2394544b9b8ffe0
- old summary.json: aba246f8cf156f16f5271a1160a63ede4475d87dbf13689665e8f6d74ad1b924

Notebook source directories are historical **remote paths**, not local access:
`/root/private_data/gnn_projects/role_mt_v1_2_export_20261018T091055Z`
and `/root/private_data/gnn_projects/role_mt_v1_2_short_20261018T091510Z`.

Minimum transfer bundle, resolved from actual manifests rather than guessed
case filenames:

1. Unchanged original six_states manifest and independent evaluation spec.
2. All six original converted NPZ and metadata: state, CLEAN p0, ORIGINAL next
   posterior q, canonical slots, families, types, prior, full alpha[0..1000],
   query_block, t, original array hashes and field-origin receipts.
3. Original collection.json per case with full query-byte hash; legacy baseline
   Genre reports and nested original64 bank artifact (not regenerated).
4. Original source_manifest/config JSON, original completed summary, six case
   JSON, frozen_objective and policy. Case metrics must equal pinned summary.
5. Public complete IMDB reference NPZ, checked against package pin.

NO private checkpoints, model tensors, original edge identity as guidance input,
notebook credentials or substitute files. Targets are extracted from the allowed
training reference into anonymous vector counts; reference adjacency is not
consulted while guide.apply computes a step. The public reference is used for
target preparation and evaluation, not per-slot reference labels.

`pack` performs read-only original-pin/source checks and a new migration mapping;
it copies all original artifacts unchanged and creates portable.json plus
portable_inputs.zip. Every actual payload hash is mapped to original path/hash.
Original manifest is NEVER rewritten. Original checkpoint hashes remain
HISTORICAL RECEIPTS ONLY; no claim of rechecking unavailable private weights.
This explicit offline contract deliberately does not call the old loader that
requires private weight files, does not fake those files, and does not claim
fully reauthenticated model provenance.

Consumer requires externally supplied portable.json SHA256, checks pins,
payload hashes, producer/old-summary receipts, array signatures, native arm and
6000-update final receipts. Rebuild original uniform query plan using seed
2026101701, [seed,t,61],20 blocks; hash raw ids/families/query bytes in original
order, not just counts. Check canonical slot/family/type order, exact supports,
prior/schedule, original posterior q, original independent64 bank and all old
baseline metrics before reporting new-method comparisons. Missing evidence
fails; no model predictions or replacement chain is permitted.

All output directories must be fresh, disjoint from source package, all actual
input evidence roots, portable bundle and B1 receipt directory. Paths are checked
before mkdir/controller writes. Runtime max900 seconds includes import,
verification, target/assignment/gradient/backtracking, exact evaluation,
input rehash, save and cleanup. Work stops12 seconds early for bounded cleanup.
Touch OUTPUT/STOP for safe stop. Partial panel cannot pass. If exact full
evaluation cannot finish, a separate user decision is needed; no approximation
or metric deletion is authorized here.

The portable fixture tests substitute pins ONLY inside pytest's monkeypatch
scope, label their inputs ENGINEERING and are not source evidence for real6.
Real conversion and compatibility remain unverified until the actual payloads
are provided. This package itself contains no private six-state arrays.
