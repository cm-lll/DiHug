#!/usr/bin/env bash
# Merge two same-checkpoint generation batches and compute complete paper metrics.
set -euo pipefail

source "${CONDA_SH:-/data2/lyh/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-sparse_block}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ROOT_A="${ROOT_A:-results/paper_final_context_binomial_xiaob}"
ROOT_B="${ROOT_B:-results/paper_final_context_binomial_xiaob_s8_15}"
OUT_ROOT="${OUT_ROOT:-paper_tables/frozen_main16}"
HGEN_PYTHON="${HGEN_PYTHON:-/data2/lyh/miniconda3/envs/hgen_env/bin/python}"
mkdir -p "$OUT_ROOT/targets" "$OUT_ROOT/distribution"
[[ -x "$HGEN_PYTHON" ]] || { echo "missing HGEN Python: $HGEN_PYTHON" >&2; exit 2; }

python - "$OUT_ROOT/targets" <<'PY'
import json, pathlib, sys
from sparse_diffusion.structure_energy.graph import load_typed_graph
out = pathlib.Path(sys.argv[1])
for dataset, path in (
    ("imdb", "data/IMDB_movie_subgraphs/subgraph_000"),
    ("pubmed", "data/PubMed_baseline_subgraphs/subgraph_000"),
):
    graph = load_typed_graph(pathlib.Path(path), active_only=True)
    (out / f"{dataset}.json").write_text(
        json.dumps(graph.to_json_dict(), indent=2, sort_keys=True)
    )
PY

for dataset in imdb pubmed; do
  for mode in legacy context; do
    generated=()
    for root in "$ROOT_A" "$ROOT_B"; do
      while IFS= read -r path; do generated+=("$path"); done < <(
        find "$root/$dataset" -path "*/$mode/final_graph.json" -type f | sort -V
      )
    done
    [[ "${#generated[@]}" -eq 16 ]] || {
      echo "expected 16 graphs for $dataset/$mode, found ${#generated[@]}" >&2
      exit 2
    }
    "$HGEN_PYTHON" scripts/evaluate_hgen_edge_split_metrics.py \
      --target "$OUT_ROOT/targets/$dataset.json" --generated "${generated[@]}" \
      --output "$OUT_ROOT/distribution/${dataset}_${mode}.json" > /dev/null
  done
done

python - "$ROOT_A" "$ROOT_B" "$OUT_ROOT" "$(git rev-parse HEAD)" <<'PY'
import csv, json, math, pathlib, statistics, sys

roots = [pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])]
out = pathlib.Path(sys.argv[3])
commit = sys.argv[4]
rows = []
seen = set()
for root in roots:
    for path in sorted(root.glob("*/seed*/*/gate_report.json")):
        dataset = path.parts[-4]
        seed = int(path.parts[-3].removeprefix("seed"))
        mode = path.parts[-2]
        if mode not in {"legacy", "context"}:
            continue
        key = (dataset, seed, mode)
        if key in seen:
            raise RuntimeError(f"duplicate run: {key}")
        seen.add(key)
        report = json.loads(path.read_text())
        if report["initial_graph_source"] != "binomial_family_count_prior":
            raise RuntimeError(f"non-Binomial report: {path}")
        row = {"dataset": dataset, "seed": seed, "mode": mode}
        for prefix in ("initial", "final", "real"):
            for metric in (
                "edges", "lcc", "triangles", "clustering", "transitivity",
                "assortativity", "sum_degree_sq", "max_degree",
            ):
                row[f"{prefix}_{metric}"] = report[prefix][metric]
        rows.append(row)

for dataset in ("imdb", "pubmed"):
    for seed in range(16):
        pair = [r for r in rows if r["dataset"] == dataset and r["seed"] == seed]
        if len(pair) != 2:
            raise RuntimeError(f"incomplete pair: {dataset}/seed{seed}")
        initial = [{k: v for k, v in r.items() if k.startswith("initial_")} for r in pair]
        if initial[0] != initial[1]:
            raise RuntimeError(f"paired initial mismatch: {dataset}/seed{seed}")

with (out / "main16_runs.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(rows)

summary = []
for dataset in ("imdb", "pubmed"):
    for mode in ("legacy", "context"):
        group = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
        dist = json.loads((out / "distribution" / f"{dataset}_{mode}.json").read_text())
        item = {"dataset": dataset, "mode": mode, "n": len(group)}
        for metric in (
            "edges", "lcc", "triangles", "clustering", "transitivity",
            "assortativity", "sum_degree_sq", "max_degree",
        ):
            values = [float(r[f"final_{metric}"]) for r in group]
            std = statistics.stdev(values)
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = std
            item[f"{metric}_ci95"] = 1.96 * std / math.sqrt(len(values))
        for metric in (
            "degree_mmd", "powerlaw_alpha", "edge_overlap_percent",
        ):
            item[f"{metric}_mean"] = dist["mean"][metric]
            item[f"{metric}_std"] = dist["std"][metric]
        item["uniqueness_percent"] = dist["uniqueness_percent"]
        summary.append(item)

with (out / "main16.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
    writer.writeheader(); writer.writerows(summary)

paired = []
for dataset in ("imdb", "pubmed"):
    for metric in ("triangles", "clustering", "lcc", "assortativity"):
        deltas = []
        for seed in range(16):
            legacy = next(r for r in rows if (r["dataset"], r["seed"], r["mode"]) == (dataset, seed, "legacy"))
            context = next(r for r in rows if (r["dataset"], r["seed"], r["mode"]) == (dataset, seed, "context"))
            deltas.append(float(context[f"final_{metric}"]) - float(legacy[f"final_{metric}"]))
        std = statistics.stdev(deltas)
        paired.append({
            "dataset": dataset, "metric": metric, "n": len(deltas),
            "context_minus_legacy_mean": statistics.fmean(deltas),
            "context_minus_legacy_std": std,
            "context_minus_legacy_ci95": 1.96 * std / math.sqrt(len(deltas)),
        })
(out / "main16.json").write_text(json.dumps({
    "format": "dihug-frozen-main16-v1", "commit": commit,
    "family_count_mode": "binomial", "paired_initials_verified": True,
    "rows": summary, "paired_differences": paired,
}, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[MAIN16] complete: $OUT_ROOT/main16.tsv"
