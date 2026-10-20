# Original C versus late-fusion C: completed paired panel

Execution source: b4b2f82916eeb94b41f7132d797897c98882eb79, based on
1613be145b095f238a805b6a9513561191ef3bd1. Source/config frozen before science.
The late readout improves sensitivity but **does not recover extreme-role control**.
No further experiment is launched.

## Scope and pairing

3 training seeds 2026102011/12/13, original C and late C, each 1000 effective
updates. Original 34-node graph panel, 8 layers, FP32, T64/R4, original CE and
AdamW settings. Final1000 only; no checkpoint selection. Sampling seeds
2026102031..38 at each balanced/extreme condition:96 actual full chains.
Common parameters were copied tensor-by-tensor and verified, separate fresh
optimizers used. All12 actual initial/final weight files remain in the persistent
local run; public SHA256 receipts are in evidence/weight_hashes.json and
weight_storage.json. No private large-graph weights are used or published.
Training input bytes and RNG, sampling prior/query/uniform/RNG pairing were
independently audited, not inferred from equal seed labels.

## Actual graph results

Each entry below is mean of8 graphs; arrows denote original→late.
W1 compares integer Genre degree distributions; TV compares complete anonymous
joint role frequencies. Genre has one active relation. Lower error is better.

|Training seed|Condition|Genre W1|Genre joint TV|Genre maximum degree mean|
|---|---|---|---|---|
|2026102011|balanced|1.5833→1.7292|.8125→.8542|4.875→5.625|
|2026102012|balanced|1.4167→1.4792|.7708→.8125|6.750→6.750|
|2026102013|balanced|1.3958→1.2708|.8333→.7500|6.625→6.250|
|2026102011|extreme|3.3750→3.4375|.7500→.7083|4.875→5.750|
|2026102012|extreme|4.3750→4.2708|.9167→.8750|6.750→6.750|
|2026102013|extreme|4.4792→4.4792|.9583→.9375|6.625→6.250|

Equal training-seed macro W1: balanced1.46528→1.49306; extreme4.07639→4.06250.
Extreme paired differences +.06250,−.10417,0 are inconsistent in direction.
All96 graphs have zero Genre nodes of degree>=12, hence also no multiple hubs.
Their largest Genre degree is9, versus target18. This is missing the desired
hub, not evidence that the method successfully controls hub multiplicity.
Genre degree<=1 mean is.41667 for both arms/conditions; extreme target requires4.

|Condition|Mean edges|Edge MAE (target88)|DMA closure (target.44444)|Connected /24|LCC|Triangles (target16)|
|---|---|---|---|---|---|---|
|balanced|80.7917→81.5417|14.7083→11.7083|.36171→.38364|14→16|33.4583→33.5833|12.0833→12.1250|
|extreme|80.7917→81.6250|14.7083→11.6250|.36171→.38364|14→16|33.4583→33.6250|12.0833→12.1250|

Other type joint-TV macro Director .88542→.88542, Actor .85417→.81250,
Movie .93981→.94213: mixed. All per-type/per-relation and per-graph data are
retained in generation.json. Each of12 seed/arm/condition cells has8 distinct
typed topologies, frequency[1,1,1,1,1,1,1,1], effective count8, no unresolved
isomorphism. These are small samples from3 training repeats, not96 independent
trained replicates or a global coverage result.

Holding sampling randomness fixed while switching balanced/extreme targets:
original0/24 pairs change graph; late1/24 (seed11), with that pair nonisomorphic.
The other23 late pairs are identical. Actual control remains weak.

## Trained fixed-input response

At identical heldout Xt/query, t1/32/64, maximum Movie–Genre p0 differences:

|Training seed|Original|Late|
|---|---|---|
|2026102011|2.44379e-6|3.38554e-4|
|2026102012|1.42157e-5|1.00198e-3|
|2026102013|1.57356e-5|4.76062e-4|

Same-condition repeated forward difference0. Stronger response is measurable,
but is not evidence of effective discrete-tail control. Final fixed validation
CE is .208833→.203764, .200853→.200926, .200843→.200244. Full0/250/500/1000
records and all6000 update losses are archived, without selecting a best point.

## Checks, costs and failure record

81 source tests passed,8 warnings,zero skips. Independent unzip:313 payload hashes,
CRC, entry help and another81 checks passed (21.32s,8warnings,zero skips).
The first preflight had a Python test-import collision, before science. Fixed
with an explicit module import; failed receipt/logs are preserved. Continued in
the SAME directory and SAME deadline with no new budget. No scientific retry,
failed chain, dropped graph or unreported isomorphism timeout.

Single activation2026-10-20 16:38:31.239992 Beijing, deadline17:38:31.239992.
Core execution and audit2602.47s; archive+unpack acceptance2827.87s cumulative.
Worker wall2501.73s/CPU2501.66s, peakRSS749.75MiB. Independent audit wall6.59s/
CPU6.55s. Disposable cost gate used10 engineering updates outside the6000 science
updates and counted within the original budget. No GPU, BW, notebook, big-graph
training, postprocessing or role-guidance sampling.
Environment Python3.9.23, Torch2.1.0 CUDA build (CPU execution), PyG2.3.1,
NumPy1.26.4, SciPy1.11.0, NetworkX2.8.7, pytest7.4.4; one CPU thread.

## Review artifacts and reproduction boundary

ZIP role_late_paired_v1_review.zip SHA256:
43022c03305c2f6f53eece58a4a80d6edecdc58539d90b2883f4c10b3c465570

It contains complete needed code, source binding, config, all raw graphs,
curves, per-seed outcomes, weight hashes and original failure/budget receipts.
Unpack verification logs/receipt and source patch are adjacent public artifacts.
Original actual weights are retained locally (paths in weight_storage.json).
Do not rerun the completed run or create another budget without authorization.
The next decision is review of these negative/mixed results, not an automatic
extension to a larger graph or longer training.
