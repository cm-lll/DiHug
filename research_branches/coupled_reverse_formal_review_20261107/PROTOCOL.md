# Frozen coupled reverse structural-risk development screen

Authorized 2026-11-07: four models, 3000 updates each, sixteen full T1000 chains.
Clock conservatively begins 10:55 BJT. Science stops 13:25, delivery 13:55;
8 GPU hours includes engineering/loading/wait/save, first applicable bound wins.

IMDB seeds 2026110411/2026110412; PubMed 2026110421/2026110422.
Sampling seeds 2026102821/2026102822/2026110231/2026110232.
Original A late CN, normal1, IID time, AdamW 2e-4, EMA .99, R1 unchanged.
Loss is CE + PB + .013715673231930771 * coupled risk / max(A,1),
A = sum_e,c (K1-K0)_e E_teacher[C_ec^2]. Formula/source copies are hash-bound.
Only this frozen loss changes relative to reused PB four-model controls.
No coefficient search, best-checkpoint selection, early performance culling,
network change, inference correction, or follow-on experiment.

Engineering: two datasets, update1/save/update2/reload/update2 (six disposable
updates), frozen-source/input/gradient/RNG checks, raw/EMA validation transaction,
two complete reverse timesteps and sampling state serialization. Temporary
models discarded. Exact full-state loading is required. Historical ordinary
GPU recompute tolerance remains failed, not weakened or claimed bitwise exact.
No short-term quality gate. Measured combined update plus historical chain costs
must fit all four models and all sixteen chains before launch.

Primary results: typed triangle count absolute error, incident4 degree W1,
joint-degree TV. Preserve all original metrics, relation errors, triangle and
MP2/MP3 compositions, Chemical low-degree/hub, connectivity and diversity.
Raw graphs primary, LCC separate; per-model four-sample mean/SD first, then
paired per-training-seed and across-two-training-seed mean/SD. Improvement in
triangles with a substantial degree/connectivity cost is a tradeoff, not a
successful overall repair. These reused development seeds are not independent
confirmation. Ordinary CE controls remain background in original reports.

Assets and parent code remain unchanged. Supervisor/watchdog owns only this
ticket's PIDs. Science deadline and GPU-hour limit independently enforced.
