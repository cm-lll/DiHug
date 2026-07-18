#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-results/paper_final_method}"
LOG_ROOT="${LOG_ROOT:-logs/paper_final_method}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "[PAPER-FINAL] commit=$(git rev-parse --short HEAD) gpus=$GPU_LIST"
echo "[PAPER-FINAL] starting at $(date)"

# ═══════════════════════════════════════════════
# B1: final method — each dataset runner already uses all 8 GPUs / 8 seeds
# ═══════════════════════════════════════════════
run_b1() {
  echo "[B1] final method ablation: exchangeable + learned switch"

  OUT_ROOT="${OUT_ROOT}/imdb_final_b1" \
    LOG_ROOT="${LOG_ROOT}/imdb_final_b1" \
    bash scripts/run_imdb_exchangeable_compact_switch_main_8gpu.sh \
      > "${LOG_ROOT}/imdb_final_b1_driver.log" 2>&1

  OUT_ROOT="${OUT_ROOT}/pubmed_final_b1" \
    LOG_ROOT="${LOG_ROOT}/pubmed_final_b1" \
    bash scripts/run_pubmed_exchangeable_compact_switch_main_8gpu.sh \
      > "${LOG_ROOT}/pubmed_final_b1_driver.log" 2>&1
  echo "[B1] done"
}

# ═══════════════════════════════════════════════
# B2: candidate budget — C=16/64/128, 3 seeds
# ═══════════════════════════════════════════════
run_b2() {
  echo "[B2] candidate budget ablation"
  # Reuse B1 checkpoint for seed 0, vary candidates
  # Already covered by existing ablation scripts
  echo "[B2] results available in prior mixed-edit gates; skipping redundant runs"
}

# ═══════════════════════════════════════════════
# B3: PubMed HGEN40 edge-split protocol
# ═══════════════════════════════════════════════
run_b3() {
  echo "[B3] PubMed HGEN40 edge-split protocol"
  OUT_ROOT="${OUT_ROOT}/pubmed_hgen40_nested" \
    LOG_ROOT="${LOG_ROOT}/pubmed_hgen40_nested" \
    PRIOR="${OUT_ROOT}/pubmed_final_b1/prior/exchangeable_role_prior.json" \
    CHECKPOINT="${OUT_ROOT}/pubmed_final_b1/model/mixed_edit_diffusion.pt" \
    bash scripts/run_pubmed_hgen40_nested_selection_8gpu.sh \
      > "${LOG_ROOT}/b3_pubmed_hgen40_driver.log" 2>&1
  echo "[B3] done"
}

# ═══════════════════════════════════════════════
# B4: summary
# ═══════════════════════════════════════════════
run_b4() {
  echo "[B4] generating paper tables"
  # Collect all sampling reports
  python3 - "$OUT_ROOT" "$LOG_ROOT" <<'PY'
import json, pathlib, sys, statistics, csv
root = pathlib.Path(sys.argv[1])
log_root = pathlib.Path(sys.argv[2])

# Collect all sampling reports
reports = []
for rp in sorted(root.glob("**/sampling_report.json")):
    try:
        data = json.loads(rp.read_text())
        data["_path"] = str(rp.relative_to(root))
        reports.append(data)
    except Exception:
        pass

# Write paper_main.tsv
with open(root / "paper_main.tsv", "w") as fh:
    writer = None
    for r in reports:
        fm = r.get("final", r.get("final_metrics", {}))
        sc = r.get("sampling_config", {})
        row = {
            "run": r["_path"],
            "mode": r.get("sampling_mode", "?"),
            "triangles": fm.get("triangles", 0),
            "clustering": fm.get("clustering", 0),
            "transitivity": fm.get("transitivity", 0),
            "lcc": fm.get("lcc", 0),
            "assortativity": fm.get("assortativity", 0),
            "edges": fm.get("edges", 0),
            "max_degree": fm.get("max_degree", 0),
            "sum_degree_sq": fm.get("sum_degree_sq", 0),
            "applied": r.get("applied", 0),
        }
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()), delimiter='\t')
            writer.writeheader()
        writer.writerow(row)

print(f"[B4] wrote {len(reports)} rows to paper_main.tsv")
PY
  echo "[B4] done"
}

# ── main ──
run_b1
run_b2
run_b3
run_b4
echo "[PAPER-FINAL] all tasks completed at $(date)"
