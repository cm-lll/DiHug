# Minimal NVIDIA recovery check (2026-11-04)

Scientific/runtime source is **175373c83202b6c8b80aa570e0092c4652031ce6**.
This follow-up adds engineering checks only. All 90 production Python files
match that commit; no model, CE, AdamW, EMA, noise kernel or sampling formula
has changed. The original execution tree was not patched or reused for a run.

## Reproduce on one approved idle GPU

Use the existing Torch 2.1.0 + CUDA 12.1 environment and the dependencies in the
source snapshot. Do not install or change packages for this check. Set a new
output directory on a data volume with space, with a `temporary` subdirectory.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
python tests/check_runtime_bindings.py --output "$CHECK_OUT/runtime_bindings.json"

CUDA_VISIBLE_DEVICES="$IDLE_GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$CHECK_OUT/temporary" \
python tests/gpu_recovery_check.py --output "$CHECK_OUT"
```

The three worker modules are imported separately without calling `main()` or
initializing CUDA. Their actual `single_input`, `sampling`, `integration` and
boundary imports are recorded. This is not a new formal launcher/approval.

## Observed result

GPU-Node01, one RTX2080Ti, driver 530.30.02, Torch 2.1.0+cu121, PyG 2.3.1,
strict deterministic algorithms with errors enabled, TF32 disabled, one CPU
thread. The same settings are already used by the actual workers.

Thirty diagnostic records passed, covering real O/A/D/E eight-layer model
layouts on a synthetic 12-node graph (66 legal slots). E uses a synthetic
engineering dictionary with the real role dimensions. No scientific dataset,
trained weight or fitted formal dictionary was loaded.

- Normal updates match the original pre-recovery update implementation in
  parameters, AdamW, EMA, full RNG, counters and inputs.
- Input/forward/nonfinite-gradient failures before optimizer start roll back
  completely. Saved/reloaded states reproduce the same next update exactly.
- Ordinary raw/EMA validation and validation-function failures preserve the
  certified training boundary, including CUDA RNG. Partial raw restore makes
  both boundary and direct serializer refuse resumable checkpoints.
- A/E use the actual reverse kernel and model on CUDA for t500 to t499.
  A failure after random draws/prediction and before outer commit restores
  graph/time/full RNG; reloaded retry equals the uninterrupted step. This is
  **not a full GPU T1000 chain**. The earlier full-chain CPU tests remain valid.

An initial attempt failed because the test harness reused an in-memory
AdamW snapshot whose CPU step tensor was aliased by `load_state_dict` and then
mutated. An independent small CPU example reproduced the fixture aliasing.
Each replay now receives an independent snapshot copy, equivalent to a fresh
disk load. The first failure is retained; production source was not changed.
The completed run took 126.060 seconds; the first attempt 14.844 seconds.

## Synchronized maintenance overhead

Each cell is the median of four paired repeats after one warmup pair, with
order alternated. Snapshot loading is outside timing. Validation contains two
real forward probes (raw+EMA). Sampling is one frozen reverse step. These are
small-graph measurements, not production-throughput estimates.

| Arm/component | Baseline ms | With transaction ms | Difference ms | Extra peak allocated MiB |
|---|---:|---:|---:|---:|
| A/update | 528.55 | 533.44 | +4.89 | 0.0005 |
| E/update | 449.08 | 463.66 | +14.58 | 0.1455 |
| A/raw+EMA validation | 462.77 | 543.42 | +80.65 | 0 |
| E/raw+EMA validation | 400.32 | 470.94 | +70.62 | 0 |
| A/reverse step | 147.33 | 147.26 | -0.07 | 0 |
| E/reverse step | 127.35 | 127.69 | +0.34 | 0 |

The tiny negative difference is timing variation, not a claimed speedup.
Benchmark allocator peaks were at most 272.56 MiB allocated / 326 MiB reserved;
these exclude CUDA driver/context memory (one system observation was 1243 MiB).
Host copies, especially raw validation snapshots, still have CPU RAM costs.

## Limits and deployment

Python exceptions were injected after known synchronization/check boundaries.
No device-side assert, illegal memory access, lost GPU/context, driver reset,
asynchronous kernel failure, OOM during partial writes, abrupt power loss or
filesystem corruption was simulated. No claim covers all CUDA failures or
bitwise determinism on arbitrary devices. Unknown partial mutation remains
non-resumable and requires an independently complete older checkpoint.

The old O-to-A experiment is already complete: four O models at 3000 updates,
eight new chains and eight reused A chains, all evaluation successful, no
restart. Its original code/configuration and outputs remain unchanged. Future
formal execution needs fresh asset paths, source hashes and approval bindings;
the old approval must not be reused with the new source. This check does not
authorize or start any formal science.
