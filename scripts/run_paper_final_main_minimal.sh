#!/usr/bin/env bash
# Minimal final evidence — prior + legacy MLP, IMDB + PubMed, 8 seeds
set -euo pipefail

source /data2/lyh/miniconda3/etc/profile.d/conda.sh
conda activate sparse_block
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-7}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
OUT_ROOT="${OUT_ROOT:-results/paper_final_main_0717}"
LOG_ROOT="${LOG_ROOT:-logs/paper_final_main_0717}"
STEPS="${STEPS:-20000}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

IMDB_PRIOR="results/imdb_exchangeable_role_prior_gate_v2/exchangeable_role_prior.json"
IMDB_CKPT="results/imdb_exchangeable_compact_s0_3/model/mixed_edit_diffusion.pt"
PUBMED_PRIOR="results/pubmed_exchangeable_compact_switch_gate_0716/prior/exchangeable_role_prior.json"
PUBMED_CKPT="results/pubmed_role_corruption_gate_0716/f100_all/model/mixed_edit_diffusion.pt"

run_dataset() {
  local name="$1" prior="$2" ckpt="$3"
  echo "[PAPER] $name seeds=${SEED_START}-${SEED_END} steps=$STEPS"
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    local gpu="${GPUS[$((seed % NGPU))]}"
    local base="$OUT_ROOT/${name}/seed${seed}"
    mkdir -p "$base"
    
    # prior-only (0 edit)
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --prior "$prior" --checkpoint "$ckpt" \
      --output-dir "$base/prior" --seed "$seed" --device cuda \
      --sample-steps 0 --sample-candidates 16 \
      > "$LOG_ROOT/${name}_s${seed}_prior.log" 2>&1 &
    
    # legacy MLP
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_exchangeable_role_prior_gate.py \
      --mode sample --prior "$prior" --checkpoint "$ckpt" \
      --output-dir "$base/legacy" --seed "$seed" --device cuda \
      --sample-steps "$STEPS" --sample-candidates 16 \
      --metrics-every "$(( STEPS / 10 ))" \
      > "$LOG_ROOT/${name}_s${seed}_legacy.log" 2>&1 &
    
    sleep 3
    wait
  done
}

run_dataset imdb "$IMDB_PRIOR" "$IMDB_CKPT"
run_dataset pubmed "$PUBMED_PRIOR" "$PUBMED_CKPT"

# Summary
python3 - "$OUT_ROOT" <<'PY'
import json,pathlib,statistics,csv,sys
root=pathlib.Path(sys.argv[1])
rows=[]
for rp in sorted(root.glob("**/sampling_report.json")):
    r=json.loads(rp.read_text())
    fm=r.get("final_metrics",{})
    rows.append({"path":str(rp.relative_to(root)),"tri":fm.get("triangles",0),
                 "clust":fm.get("clustering",0),"lcc":fm.get("lcc",0),
                 "edges":fm.get("edges",0),"applied":r.get("applied",0)})
with open(root/"paper_main.tsv","w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),delimiter='\t')
    w.writeheader(); w.writerows(rows)
groups={}
for r in rows:
    parts=r["path"].split("/")
    g="/".join(parts[:2])
    groups.setdefault(g,[]).append(r)
for g,items in sorted(groups.items()):
    tris=[i["tri"] for i in items]
    print(f"{g}: n={len(items)} tri={statistics.mean(tris):.0f}±{statistics.stdev(tris) if len(tris)>1 else 0:.0f}")
PY
echo "[PAPER] done: $OUT_ROOT/paper_main.tsv"
