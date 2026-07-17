#!/usr/bin/env bash
# DiHug final evidence pipeline — frozen method, paired experiments, auto-summary
set -euo pipefail

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ── config ──
COMMIT="$(git rev-parse --short HEAD)"
MACHINE_TAG="${MACHINE_TAG:-local}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
OUT_ROOT="${OUT_ROOT:-results/paper_final_evidence_${MACHINE_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_final_evidence_${MACHINE_TAG}}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
IMDB_DIR="data/IMDB_movie_subgraphs/subgraph_000"
PUBMED_DIR="data/PubMed_baseline_subgraphs/subgraph_000"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# ── checkpoint paths (pre-trained by exchangeable compact switch) ──
IMDB_LEGACY="results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt"
PUBMED_LEGACY="results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt"
IMDB_CTX="results/rooted_context_freeze/imdb/rooted_context_residual.pt"
PUBMED_CTX="results/rooted_context_freeze/pubmed/rooted_context_residual.pt"

echo "[FINAL-EVIDENCE] commit=$COMMIT machine=$MACHINE_TAG seeds=$SEED_START-$SEED_END gpus=$GPU_LIST"
echo "[FINAL-EVIDENCE] started at $(date)"

# ═══════════════════════════════════════════════════════
# Stage 1: full-graph main results (paired, 8 seeds)
# ═══════════════════════════════════════════════════════
run_main() {
  local dataset="$1" legacy_ckpt="$2" ctx_ckpt="$3" steps="$4"
  local name tag dir
  name="${dataset}"
  dir="$(dirname "$legacy_ckpt")"
  
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    local gpu="${GPUS[$((seed % NGPU))]}"
    local base="$OUT_ROOT/${name}/seed${seed}"
    mkdir -p "$base"
    
    for mode in prior random legacy context; do
      tag="${name}_s${seed}_${mode}"
      case "$mode" in
        prior)
          # A: learned prior only, 0 edit
          CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
            --mode sample --prior "$dir/exchangeable_role_prior.json" \
            --output-dir "$base/prior" --seed "$seed" --device cuda \
            --sample-steps 0 --sample-candidates 16 \
            > "$LOG_ROOT/${tag}.log" 2>&1 &
          ;;
        random)
          # B: random legal edits
          CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
            --mode sample --prior "$dir/exchangeable_role_prior.json" \
            --checkpoint "$legacy_ckpt" --output-dir "$base/random" \
            --seed "$seed" --device cuda --sample-steps "$steps" \
            --sample-candidates 16 --sampling-mode random \
            > "$LOG_ROOT/${tag}.log" 2>&1 &
          ;;
        legacy)
          # C: legacy listwise MLP
          CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
            --mode sample --prior "$dir/exchangeable_role_prior.json" \
            --checkpoint "$legacy_ckpt" --output-dir "$base/legacy" \
            --seed "$seed" --device cuda --sample-steps "$steps" \
            --sample-candidates 16 --sampling-mode model \
            > "$LOG_ROOT/${tag}.log" 2>&1 &
          ;;
        context)
          # D: MLP + rooted-context residual
          CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_rooted_context_residual_gate.py \
            --mode sample --dataset-dir "$(dataset_path "$name")" \
            --legacy-checkpoint "$legacy_ckpt" \
            --context-checkpoint "$ctx_ckpt" \
            --output-dir "$base/context" --seed "$seed" --device cuda \
            --steps "$steps" --candidates 16 --metrics-every "$(( steps / 10 ))" \
            > "$LOG_ROOT/${tag}.log" 2>&1 &
          ;;
      esac
      # stagger per seed to avoid driver contention
      sleep 2
    done
    wait
  done
}

dataset_path() {
  case "$1" in imdb) echo "$IMDB_DIR" ;; pubmed) echo "$PUBMED_DIR" ;; *) echo "?" ;; esac
}

echo "[FINAL-EVIDENCE] Stage 1: IMDB 20k steps" | tee -a "$LOG_ROOT/driver.log"
run_main imdb "$IMDB_LEGACY" "$IMDB_CTX" 20000

echo "[FINAL-EVIDENCE] Stage 1: PubMed 20k steps" | tee -a "$LOG_ROOT/driver.log"
run_main pubmed "$PUBMED_LEGACY" "$PUBMED_CTX" 20000

# ═══════════════════════════════════════════════════════
# Stage 2: core ablations (candidate count, budget, kernel)
# ═══════════════════════════════════════════════════════
echo "[FINAL-EVIDENCE] Stage 2: ablation (candidate counts)" | tee -a "$LOG_ROOT/driver.log"
for candidates in 8 16 32; do
  for seed in 0 1 2 3; do
    gpu="${GPUS[$((seed % NGPU))]}"
    base="$OUT_ROOT/ablation/c${candidates}/seed${seed}"
    mkdir -p "$base"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --prior "$(dirname "$IMDB_LEGACY")/exchangeable_role_prior.json" \
      --checkpoint "$IMDB_LEGACY" --output-dir "$base" \
      --seed "$seed" --device cuda --sample-steps 20000 \
      --sample-candidates "$candidates" --sampling-mode model \
      > "$LOG_ROOT/ablation_c${candidates}_s${seed}.log" 2>&1 &
    sleep 2
  done
  wait
done

# ═══════════════════════════════════════════════════════
# Stage 3: summary generation
# ═══════════════════════════════════════════════════════
echo "[FINAL-EVIDENCE] Stage 3: summary" | tee -a "$LOG_ROOT/driver.log"
python3 - "$OUT_ROOT" "$COMMIT" "$MACHINE_TAG" <<'PY'
import json, pathlib, sys, statistics, csv

root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
machine = sys.argv[3]

def load_metrics(report_path):
    try:
        r = json.loads(report_path.read_text())
        fm = r.get("final_metrics", r.get("metrics", {}))
        im = r.get("initial_metrics", r.get("initial", fm))
        sm = r.get("sampling_config", {})
        return {
            "path": str(report_path.relative_to(root)),
            "mode": r.get("sampling_mode", r.get("mode", "?")),
            "applied": r.get("applied", 0),
            "tri": fm.get("triangles", 0),
            "clust": fm.get("clustering", 0),
            "trans": fm.get("transitivity", 0),
            "lcc": fm.get("lcc", 0),
            "edges": fm.get("edges", 0),
            "assort": fm.get("assortativity", 0),
            "max_deg": fm.get("max_degree", 0),
            "sum_deg_sq": fm.get("sum_degree_sq", 0),
            "init_tri": im.get("triangles", 0),
            "init_clust": im.get("clustering", 0),
            "init_lcc": im.get("lcc", 0),
            "init_edges": im.get("edges", 0),
            "candidates": sm.get("candidates_per_step", sm.get("sample_candidates", "?")),
            "steps": sm.get("requested_steps", sm.get("sample_steps", "?")),
        }
    except Exception:
        return None

rows = []
for rp in sorted(root.glob("**/sampling_report.json")):
    row = load_metrics(rp)
    if row:
        rows.append(row)

# Write main TSV
with open(root / "paper_main.tsv", "w") as fh:
    if rows:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader()
        w.writerows(rows)

# Aggregate by experiment group
groups = {}
for r in rows:
    parts = r["path"].split("/")
    group = "/".join(parts[:3]) if len(parts) >= 3 else r["path"]
    groups.setdefault(group, []).append(r)

with open(root / "paper_aggregate.tsv", "w") as fh:
    fh.write("group\tn\ttri_mean\ttri_std\tclust_mean\tclust_std\tlcc_mean\tmode\n")
    for group, items in sorted(groups.items()):
        tris = [i["tri"] for i in items]
        clusts = [i["clust"] for i in items]
        lccs = [i["lcc"] for i in items]
        mode = items[0]["mode"] if items else "?"
        fh.write(f"{group}\t{len(items)}\t{statistics.mean(tris):.1f}\t{statistics.stdev(tris) if len(tris)>1 else 0:.1f}\t{statistics.mean(clusts):.4f}\t{statistics.stdev(clusts) if len(clusts)>1 else 0:.4f}\t{statistics.mean(lccs):.1f}\t{mode}\n")

# Write metadata
meta = {
    "commit": commit,
    "machine": machine,
    "total_runs": len(rows),
    "datasets": sorted(set(r["path"].split("/")[0] for r in rows)),
}
with open(root / "paper_metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)

print(f"[SUMMARY] {len(rows)} runs, {len(groups)} groups")
for g, items in sorted(groups.items()):
    tris = [i["tri"] for i in items]
    print(f"  {g}: n={len(items)} tri={statistics.mean(tris):.0f}±{statistics.stdev(tris) if len(tris)>1 else 0:.0f}")
PY

echo "[FINAL-EVIDENCE] all stages completed at $(date)" | tee -a "$LOG_ROOT/driver.log"
echo "[FINAL-EVIDENCE] results in $OUT_ROOT/paper_main.tsv $OUT_ROOT/paper_aggregate.tsv"
