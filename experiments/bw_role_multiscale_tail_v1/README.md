# Independent multiscale joint-role CF + analytic tail guidance

Base: fixed `acbbddb5cdbcbaa2ae129bcd59e434e6a298b130`.
Independent implementation using our existing v1 only; no other-assistant
source or ZIP consumed. All64 original package files except the bundle-wide
SHA256SUMS are byte-identical to the fixed base.

Read `package/MULTISCALE_ROLE_DESIGN.md` and `package/INPUT_CONTRACT.md`.
New code: `package/multiscale_role/`; use `package/multiscale.sh`, not the
retained historical launchers. Core additive patch:
`package/stepwise_v1_to_multiscale_tail_v1.diff`.

Frozen objective: three scales ×32 reference-derived directions, normalized
joint-CF loss and exact analytic thinned-tail loss, weights .5/.5. Existing
posterior, sampling controller, noise, network and training loss unchanged.
Genre referenceD90=153, tail probe p=1/51; total/relation duplicate removed.
No six-state outcomes or independent64 bank used to select any setting.

Original59 +new26 =85 CPU regressions passed. Fresh release extraction also
passed85. Missing-real-input invocation exits2 with protocol_failed and no
model calls. Read review_acceptance.json for exact release-hash acceptance.

Real A/B t500/20/1 input probabilities/states, original Genre64 artifacts and
historical W1 definition are not available locally. Six-state short test and
BW/GPU measurements NOT executed. This is no effectiveness or performance win.
Only CPU reference is implemented; no GPU acceleration claimed.

ZIP/checksum are in release/. No private weights or real-state probabilities
are uploaded. No training, full scientific chain or successor automatically
started. Await code review before any manual six-state invocation (<=900s).
