# Frozen local paired C experiment

Base1613be145b095f238a805b6a9513561191ef3bd1. Original C versus late-fusion C,
both from fresh initialization; never load an old trained model. Model modules
are imported unchanged from the prior independent package. Original34node panel,
full8layer FP32, AdamW and stable CE unchanged,T64/R4. No clipping, EMA, scheduler,
guidance, repair or extra scientific model. CPU one thread.

3original training seeds2026102011/12/13;1000effective updates per arm.
Validation0/250/500/1000 on original fixed validation panel. Final weights only.
Sampling seeds2026102031..38,8per condition per arm per training seed:96chains.
Both groups are conditional models, so all96outputs are actually generated.
Shared parameters copied tensor-by-tensor; old role_out/new role_fusion differ
and start at zero. Every actual initial state saved, each optimizer independent
and empty. Every paired update uses one shared fully constructed input, with
full input digest before/after and matching Torch RNG. Initial prior states,
each complete-step uniform bytes, query bytes and RNG checked across arms and
conditions. CompleteXt frozen until all4blocks complete, unchanged sampler.

First activation creates an exclusive persistent budget ledger.3600seconds
cumulative includes frozen-source binding,81CPU tests, cost probe, scientific
updates, validation,96chains, evaluation, initial/final weights and audit.
No resume/new directory budget reset. Worker saves partial weights on catchable
stop/failure with per-arm counts; partial pairs/chains are not completed results.
Parent deadline includes all children; own process-group cleanup at end.
No notebook/BW/GPU. Source/config frozen before tests and scientific run.
Cost gate uses disposable models,5updates/arm,3dense complete steps/arm, full
validation.1.15forecast margin plus90seconds audit/IO; reject whole scope if it
does not fit, never drop seeds/conditions or change settings after seeing quality.

Actual graphs: GenreW1 and joint-role TV, maximum degree, G>=12 count and >1hub,
Gzero/degree<=1, relation counts, LCC/isolates, typed triangles and DMAclosure.
Topology diversity measured within each seed/arm/condition via true-type/
relation-preserving isomorphism;1second per pair, undecided counts reported.
Class positive identifications retain mapping evidence; WL/hash not a proof.
Undefined closure remains NA. No composite success score or new quality cutoff.
Report paired differences by training seed, balanced/extreme separately.
Responses after training use same fixedXt and full4blocks at t1/32/64,
balanced/extreme/balanced; record logit gap and p0, MG separately.

Primary interpretation: improved extreme actual-tail/role distribution with
ordinary-condition and edge/closure/connectivity/diversity tradeoffs exposed.
Response alone, loss alone, hand-weight capacity witness or one best seed cannot
establish success. Failure, no improvement, regression and incomplete scope kept.
No additional experiments after this panel.

Run exactly once after freezing commit:
```
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
 /data2/lyh/miniconda3/envs/dihug/bin/python -u run.py --output /absolute/new/output
```
Progress: tail output/progress.jsonl and launcher_progress.json. Safe stop:
touch output/STOP. This run is not resumable for extra learning/time.
