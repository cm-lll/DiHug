#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEFAULT_PYTHON="/data2/lyh/miniconda3/envs/sparse_block/bin/python"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
[[ -x "$PYTHON" ]] || {
  echo "Python not executable: $PYTHON (set PYTHON explicitly)" >&2
  exit 2
}
OUT_ROOT="${OUT_ROOT:-results/closed_form_role_gate_0716}"
LOG_ROOT="${LOG_ROOT:-logs/closed_form_role_gate_0716}"
EPOCHS="${EPOCHS:-30}"
TRAJECTORIES="${TRAJECTORIES:-32}"
CANDIDATES="${CANDIDATES:-64}"
SAMPLE_STEPS="${SAMPLE_STEPS:-2000}"
FIXED_T_TRIALS="${FIXED_T_TRIALS:-16}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/launcher.log"

run_one() {
  local gpu="$1" dataset="$2" dataset_dir="$3" variant="$4" seed="$5"
  local output="$OUT_ROOT/${dataset}_${variant}_s${seed}"
  local log="$LOG_ROOT/${dataset}_${variant}_s${seed}.log"
  local profile="typed_closure_full_v2"
  local role_args=()
  if [[ "$variant" == "iid_role" ]]; then
    profile="typed_closure_role_v3"
    role_args=(--anonymous-role-condition --role-prior-components 64)
  fi
  mkdir -p "$output"
  echo "[START] gpu=$gpu dataset=$dataset variant=$variant seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode train --dataset-dir "$dataset_dir" --output-dir "$output/train" \
    --denoiser listwise --kernel bdrs --device cuda --seed "$seed" \
    --trajectories "$TRAJECTORIES" --forward-mode closed_form \
    --closed-form-noise-fractions 0.05,0.2,0.5,0.8,1.0 \
    --train-candidates "$CANDIDATES" --epochs "$EPOCHS" --batch-size 32 \
    --hidden-dim 256 --layers 3 --training-objective utility_kl \
    --utility-temperature 1.0 --type-orbit-augmentation \
    --feature-profile "$profile" --fixed-t-trials "$FIXED_T_TRIALS" \
    "${role_args[@]}" > "$log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_mixed_edit_diffusion.py \
    --mode sample --dataset-dir "$dataset_dir" --output-dir "$output/sample" \
    --checkpoint "$output/train/mixed_edit_diffusion.pt" \
    --denoiser listwise --kernel bdrs --device cuda --seed "$seed" \
    --sample-steps "$SAMPLE_STEPS" --sample-candidates "$CANDIDATES" \
    --sample-temperature 0 --metrics-every 200 >> "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$dataset variant=$variant seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

run_one 0 imdb data/IMDB_movie_subgraphs/subgraph_000 no_role 0 &
p0=$!
sleep 5
run_one 1 imdb data/IMDB_movie_subgraphs/subgraph_000 iid_role 0 &
p1=$!
sleep 5
run_one 2 pubmed data/PubMed_baseline_subgraphs/subgraph_000 no_role 0 &
p2=$!
sleep 5
run_one 3 pubmed data/PubMed_baseline_subgraphs/subgraph_000 iid_role 0 &
p3=$!

failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "closed-form role gate failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
rows=[]
for run in sorted(root.glob('*_s0')):
    train=json.loads((run/'train/train_report.json').read_text())
    sample=json.loads((run/'sample/sampling_report.json').read_text())
    dataset=run.name.split('_',1)[0]
    variant=run.name[len(dataset)+1:-3]
    rows.append({
        'dataset':dataset,
        'variant':variant,
        'fixed_t':train['fixed_t'],
        'initial':sample['initial'],
        'final':sample['final'],
        'action_counts':sample['action_counts'],
    })
payload={
    'format':'dihug-closed-form-role-gate-v1',
    'selection_uses_final_metrics':False,
    'sampling_uses_clean_graph':False,
    'rows':rows,
}
(root/'summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps(payload,indent=2,sort_keys=True))
PY

echo "[CLOSED-FORM-ROLE-GATE] complete: $OUT_ROOT/summary.json"
