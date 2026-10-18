# Joint CN v1.1: two review fixes only

Base: fixed published commit `6379e992c107a1d9c2dd124d3e976d2aa2004f2e`.
Old package/ZIP and all prior evidence remain untouched. No architecture,
optimizer, forward-noise, query, reverse-posterior, sampling or RNG changes.

## Stable loss

For free slots, compute `cross_entropy([z_absent,z_present], hard_label)`
directly. For exact prior0/1, first validate the label agrees with support,
then use differentiable exact zero loss/gradient without evaluating their CE.
Invalid/nonfinite labels, logits or priors are rejected. Training remains
`sum(per_slot_CE) * ce_weight`, retaining the original R/S denominator, including
fixed slots in that denominator. No rebalancing, epsilon or label smoothing.

`common.update` requests logits even without diagnostics; no probability export
is added to ordinary training. `fusion_checks.diagnostics` and split training
logs use the same supported CE helper. The original probability output and
exact prior projection remain available unchanged for inference/validation of
support. This fixes FP32 probability-saturation loss gradients; it is not a new
objective in exact arithmetic on free slots and does not prove training quality.

Regressions compare moderate FP64 values/gradients with the old objective and
an analytic gradient; extreme FP32 wrong-confidence logits retain nonzero
corrective gradients. All-fixed/empty and illegal-label cases are tested.

## Decision responses

Keep the original max displacement of both logits as an interface metric.
Also measure the change in `z_present-z_absent` and supported p0. For CN×global
inputs compute the four-corner mixed response separately in logits, gap and
p0. Gap and probability responses must exceed the predeclared 1e-8 floor in
the fixed engineering fixture; no threshold was relaxed after inspection.

Counterexamples add8 to both logits. Interface displacement and a pure shared
shift interaction can be8 while both gap and probability effects are exactly0.
Actual random-weight head responses only show interface capability; they do
not show learned coordination, useful guidance or full-chain stability.

## Scope and evidence

The original27 tests remain byte-identical;12 targeted CPU cases are added.
No BW, optimizer update/formal training, real IMDB inference or scientific
generation is started for this repair. Original tiny-chain unit fixtures still
run as part of the explicitly requested original27 tests. Old architecture,
data, math kernel, query and resource guards are hash-checked against the base.
See `evidence/` for raw test XML/logs, measured resources and numeric witnesses;
`joint_cn_v1_to_v1_1.diff` is the relative patch. `DESIGN_AND_STAGED_PLAN.md`
is preserved as the original design, not a newly issued experiment.

Runtime architecture/config names intentionally remain v1; the independent
package release is v1.1 with new code hashes/manifest. Existing recovery/source
guards are not bypassed. Old checkpoints are not automatically resumed.
