#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparse_diffusion.structure_energy.graph import load_typed_graph
from sparse_diffusion.structure_energy.role_guided_edit_diffusion import (
    ExchangeableRoleDegreePrior,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the type-exchangeable anonymous role condition."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = load_typed_graph(args.dataset_dir, active_only=True)
    prior = ExchangeableRoleDegreePrior.from_training_graph(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior.save(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "format": "dihug-frozen-anonymous-role-prior-v1",
        "dataset": str(args.dataset_dir),
        "output": str(args.output),
        "sha256": digest,
        "type_names": list(prior.type_names),
        "family_names": list(prior.family_names),
        "inference_contract": {
            "clean_adjacency_access": False,
            "original_node_ids": False,
            "evaluation_metrics": False,
            "type_exchangeable_rows": True,
        },
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
