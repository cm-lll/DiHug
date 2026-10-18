# Dynamic matching v1.1: CLI and release binding only

Baseline review commit: **50c6ab1d9df245cd103fbacd42dfeb7298d417c9**.
The v1 directory, ZIP, weights, historical evidence and main are preserved.

## Changes

- `run_dynamic.py` now propagates the INTEGER returned by the unchanged
  `supervise` implementation. Status display reads `controller.json` separately;
  it never treats an exit code as a dictionary or remaps all failures to1.
- B1 and six-state receipts additionally bind the SHA256 of the complete
  `DYNAMIC_SHA256SUMS` release checklist. The public parent still checks EVERY
  listed file before spawn and after completion. This covers reused posterior,
  sampling and evaluation code as well as `dynamic_role`. Target, portable input,
  module fingerprint and Python/NumPy/SciPy bindings remain mandatory. Old B1
  receipts without the new release field are rejected, not silently grandfathered.
- Public `smoke` action is explicitly ENGINEERING ONLY: one fixed four-slot
  probability test, no model, private weights, real-state input or graph sampling.
  `invalid` supplies an illegal toy probability to exercise a genuine child
  failure. `wait` emulates a stuck child so timeout/STOP must be handled by the
  original parent supervisor. It cannot run beyond the unchanged inclusive bound.

No dynamic-matching mathematics, objective normalization, support, controller
settings, scientific policy, model/loss, noise, posterior, T1000 or query plan
changes. Old133+39 test cases remain present. Earlier evidence directories describe
the baseline runs; current evidence is in `evidence_cli_v1_1` and release receipts.

## Safe CPU command after extracting this NEW ZIP

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /usr/local/bin/python3.11 run_dynamic.py smoke \
  --output /tmp/dynamic_role_cli_smoke_01 --max-seconds 20
echo $?
```

Success requires exit0, controller status completed, child status complete and
the hash-verified `engineering_result.json`. This is not a six-state run.

```bash
python verify_dynamic.py --output /tmp/dynamic_role_v1_1_full_tests_01
```

The full test wrapper freezes the exact complete test collection and requires
zero skipped/failed/missing cases. Public CLI tests genuinely start parent,
child and supervisor for success, invalid input, timeout124 and STOP75. Tests
use only their own process groups and fresh temporary engineering outputs.
SIGKILL/machine loss remain uncatchable; no new claim is made about them.

Existing scientific `pack/check/b1/six` arguments and900-second maximum remain
unchanged. This repair does NOT authorize running those real-data commands,
BW, training or full chains. No real six-state inputs were read for this ticket.
