# Fixed-weight DP/Newton generation review

Read RESULTS.md first. All528 newly executed T64/R4 chains, raw graphs, all64-step states,
Newton likelihood/P/logP and all per-read certificates are included; no private PT weights.
evidence/freeze.json binds original three broad final6000 weights,11 targets,eight paired seeds,source and environment.
evidence/activation.json is the single2700-second ledger from first preparation, never reset.
Controller/worker receipts contain completion/failure/cost; engineering failures retained separately.
generation/per_seed/aggregate/topology_classes/trajectory_divergence/audit contain all graph and paired summaries.
Review_bundle and ZIP include complete unchanged model/kernel/DP/Newton dependencies and source patch.
Every new Newton read is checked; failure cannot silently fall back to DP.
No training,GPU/BW,large network,graph repair or additional seeds. Population diversity is not proven by eight samples.
Independent ZIP unpack/test receipt is outside the immutable archive and added to the review commit afterward.
Execution is stopped after publication, awaiting review; do not recreate the activation to run another panel.

Independent unpack verification completed:2846 payload hashes,worker --help and the exact39-test set all passed,zero skips.
Unpack audit took10.03s and used no weights/model fitting/chains. See unpacked_acceptance.json and unpacked_tests.xml/log outside the ZIP.
Archive role_newton_generation_v1_review.zip:45,982,125 bytes.
SHA256 d1606f10888b7a8799f3e5400435e1abed59ac8d7f9bce7d232cf8ebb0e52f0d.
