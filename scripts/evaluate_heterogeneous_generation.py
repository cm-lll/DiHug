#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_diffusion.structure_energy.graph import TypedGraph, load_typed_graph
from sparse_diffusion.structure_energy.heterogeneous_metrics import (
    aggregate_comparisons,
    compare_heterogeneous_graphs,
    save_comparison,
)


def load_graph(path: Path) -> TypedGraph:
    if path.is_dir():
        return load_typed_graph(path, active_only=True)
    return TypedGraph.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate full and typed structure metrics for generated graphs."
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--generated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = load_graph(args.target)
    rows = []
    for sample_id, path in enumerate(args.generated):
        row = compare_heterogeneous_graphs(target, load_graph(path))
        row["sample"] = sample_id
        row["path"] = str(path)
        rows.append(row)
    payload = {
        "format": "dihug-heterogeneous-generation-evaluation-v1",
        "target": str(args.target),
        "rows": rows,
        "aggregate": aggregate_comparisons(rows),
    }
    save_comparison(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
