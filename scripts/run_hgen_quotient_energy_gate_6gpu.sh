#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-results/hgen_quotient_energy_gate_0716}"
LOG_ROOT="${LOG_ROOT:-logs/hgen_quotient_energy_gate_0716}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$LOG_ROOT/launcher.log"

dataset_dir() {
  case "$1" in
    syn100) echo data/HGEN_syn_100_subgraphs/subgraph_000 ;;
    syn200) echo data/HGEN_syn_200_subgraphs/subgraph_000 ;;
    syn500) echo data/HGEN_syn_500_subgraphs/subgraph_000 ;;
    *) return 1 ;;
  esac
}

load_profile() {
  eval "$("$PYTHON" - "$1" <<'PY'
import json, pathlib, shlex, sys
key=sys.argv[1]
p=json.loads((pathlib.Path('results/hgen_official_adaptive_0716')/key/'adaptive_profile.json').read_text())
for k,v in {
 'SAMPLE_STEPS':p['sample_steps'], 'ROLE_COMPONENTS':p['role_components'],
 'EDGE_COUNT':p['num_edges'],
}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
}

run_one() {
  local gpu="$1" key="$2" seed="$3"
  load_profile "$key"
  local out="$OUT_ROOT/$key/seed${seed}"
  local log="$LOG_ROOT/${key}_s${seed}.log"
  local replay_steps=$(( (EDGE_COUNT + 19) / 20 ))
  mkdir -p "$out"
  echo "[START] gpu=$gpu dataset=$key seed=$seed replay_steps=$replay_steps" | tee -a "$LOG_ROOT/launcher.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_quotient_energy_gate.py \
    --mode train --dataset-dir "$(dataset_dir "$key")" \
    --prior "results/hgen_official_adaptive_0716/$key/shared_prior/exchangeable_role_prior.json" \
    --output-dir "$out/model" --device cuda --seed "$seed" \
    --trajectories 12 --noise-fractions 0.05,0.2,1.0 \
    --epochs 20 --hidden-dim 96 --heads 4 --layers 2 \
    --replay-rounds 1 --replay-steps "$replay_steps" --replay-candidates 16 \
    > "$log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_quotient_energy_gate.py \
    --mode sample --dataset-dir "$(dataset_dir "$key")" \
    --prior "results/hgen_official_adaptive_0716/$key/shared_prior/exchangeable_role_prior.json" \
    --checkpoint "$out/model/quotient_energy.pt" --output-dir "$out/sample" \
    --device cuda --seed "$seed" --sample-steps "$SAMPLE_STEPS" \
    --sample-candidates 16 --metrics-every "$((SAMPLE_STEPS / 10))" \
    --minimum-energy-drop -1 --noop-patience 100 >> "$log" 2>&1
  echo "[DONE] gpu=$gpu dataset=$key seed=$seed" | tee -a "$LOG_ROOT/launcher.log"
}

keys=(syn100 syn100 syn200 syn200 syn500 syn500)
seeds=(0 1 0 1 0 1)
pids=()
for gpu in $(seq 0 5); do
  run_one "$gpu" "${keys[$gpu]}" "${seeds[$gpu]}" &
  pids+=("$!")
  sleep 3
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" -eq 0 ]] || { echo "quotient energy gate failed" >&2; exit 1; }

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json, pathlib, statistics, sys
from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.switch_chain import graph_metrics

root=pathlib.Path(sys.argv[1])
datasets={
 'syn100':pathlib.Path('data/HGEN_syn_100_subgraphs/subgraph_000'),
 'syn200':pathlib.Path('data/HGEN_syn_200_subgraphs/subgraph_000'),
 'syn500':pathlib.Path('data/HGEN_syn_500_subgraphs/subgraph_000'),
}
rows=[]
for key,path in datasets.items():
 real=graph_metrics(load_typed_graph(path,active_only=True))
 for seed in (0,1):
  run=root/key/f'seed{seed}'
  train=json.loads((run/'model/train_report.json').read_text())
  sample=json.loads((run/'sample/sampling_report.json').read_text())
  def errors(stage):
   tri=abs(stage['triangles']-real['triangles'])/max(real['triangles'],1.0)
   clu=abs(stage['clustering']-real['clustering'])/max(real['clustering'],1e-12)
   return {'triangle':tri,'clustering':clu,'mean':(tri+clu)/2}
  rows.append({'dataset':key,'seed':seed,'real':real,'initial':sample['initial'],
   'final':sample['final'],'initial_error':errors(sample['initial']),
   'final_error':errors(sample['final']),'train':train,
   'action_counts':sample['action_counts'],'minimum_energy_drop':sample['minimum_energy_drop']})
groups={}
for key in datasets:
 selected=[r for r in rows if r['dataset']==key]
 groups[key]={
  'initial_mean_error':statistics.fmean(r['initial_error']['mean'] for r in selected),
  'final_mean_error':statistics.fmean(r['final_error']['mean'] for r in selected),
  'all_seeds_non_degrading':all(r['final_error']['mean']<=r['initial_error']['mean'] for r in selected),
  'ranking_accuracy_mean':statistics.fmean(r['train']['overall_ranking_accuracy'] for r in selected),
  'orbit_invariance_max':max(r['train']['orbit_energy_absolute_difference'] for r in selected),
 }
payload={'format':'dihug-hgen-quotient-energy-gate-v1','training_uses_final_metrics':False,
 'sampling_uses_clean_graph':False,'selection_uses_final_metrics':False,
 'passed':all(v['all_seeds_non_degrading'] and v['orbit_invariance_max']<1e-5 for v in groups.values()),
 'groups':groups,'rows':rows}
(root/'summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps(payload,indent=2,sort_keys=True))
PY

echo "[QUOTIENT-ENERGY-GATE] complete: $OUT_ROOT/summary.json"
