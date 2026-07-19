#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluate_heterogeneous_generation import load_graph
from sparse_diffusion.structure_energy.heterogeneous_metrics import (
    aggregate_comparisons,
    compare_heterogeneous_graphs,
    save_comparison,
)


DATASETS = {
    "imdb": Path("data/IMDB_movie_subgraphs/subgraph_000"),
    "pubmed": Path("data/PubMed_baseline_subgraphs/subgraph_000"),
    "dblp": Path("data/DBLP_four_area_subgraphs/subgraph_000"),
    "acm": Path("data/ACM_subgraphs/subgraph_000"),
    "syn100": Path("data/HGEN_syn_100_subgraphs/subgraph_000"),
    "syn200": Path("data/HGEN_syn_200_subgraphs/subgraph_000"),
    "syn500": Path("data/HGEN_syn_500_subgraphs/subgraph_000"),
}


def _flatten(dataset: str, mode: str, payload: dict) -> dict:
    aggregate = payload["aggregate"]
    global_metrics = aggregate["global"]
    row = {
        "dataset": dataset,
        "mode": mode,
        "samples": aggregate["samples"],
        "uniqueness_percent": aggregate["uniqueness_percent"],
        "degree_mmd": aggregate["degree_mmd"]["mean"],
        "powerlaw_target_alpha": aggregate["powerlaw"]["target"]["alpha"],
        "powerlaw_generated_alpha": aggregate["powerlaw"]["generated_alpha"]["mean"],
        "powerlaw_target_ks": aggregate["powerlaw"]["target"]["ks"],
        "powerlaw_generated_ks": aggregate["powerlaw"]["generated_ks"]["mean"],
        "typed_triangle_tv": aggregate["typed_triangles"]["tv"]["mean"],
        "typed_triangle_jsd": aggregate["typed_triangles"]["jsd"]["mean"],
        "edge_recall": aggregate["edge_identity_diagnostic"]["target_edge_recall"]["mean"],
        "edge_overlap_percent": aggregate["edge_identity_diagnostic"]["edge_overlap_percent"]["mean"],
        "edge_novelty": aggregate["edge_identity_diagnostic"]["generated_edge_novelty"]["mean"],
    }
    for metric in (
        "edges",
        "lcc",
        "triangles",
        "clustering",
        "transitivity",
        "assortativity",
        "sum_degree_sq",
        "max_degree",
    ):
        values = global_metrics[metric]
        row[f"{metric}_target"] = values["target"]
        row[f"{metric}_generated"] = values["generated"]["mean"]
        row[f"{metric}_relative_error"] = values["relative_error"]["mean"]
    for length in ("1", "2", "3"):
        row[f"typed_metapath_l{length}_tv"] = aggregate["typed_metapaths"][length]["tv"]["mean"]
        row[f"typed_metapath_l{length}_jsd"] = aggregate["typed_metapaths"][length]["jsd"]["mean"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full heterogeneous structure tables from saved graphs."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to evaluate; defaults to every dataset found under results-root.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        help="Generation mode subdirectory; defaults to every discovered mode.",
    )
    args = parser.parse_args()

    datasets = args.dataset or [
        name for name in DATASETS if (args.results_root / name).exists()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for dataset in datasets:
        target_path = DATASETS[dataset]
        if not target_path.exists():
            raise FileNotFoundError(f"missing target dataset: {target_path}")
        target = load_graph(target_path)
        seed_dirs = sorted((args.results_root / dataset).glob("seed*"))
        if not seed_dirs:
            raise FileNotFoundError(
                f"no seed directories under {args.results_root / dataset}"
            )
        modes = args.mode or sorted(
            {
                child.name
                for seed_dir in seed_dirs
                for child in seed_dir.iterdir()
                if child.is_dir()
            }
        )
        for mode in modes:
            paths = [seed_dir / mode / "final_graph.json" for seed_dir in seed_dirs]
            paths = [path for path in paths if path.exists()]
            if not paths:
                continue
            rows = []
            for path in paths:
                comparison = compare_heterogeneous_graphs(target, load_graph(path))
                comparison["seed"] = int(path.parts[-3].removeprefix("seed"))
                comparison["path"] = str(path)
                rows.append(comparison)
            payload = {
                "format": "dihug-paper-extended-structure-v1",
                "dataset": dataset,
                "mode": mode,
                "target": str(target_path),
                "results_root": str(args.results_root),
                "rows": rows,
                "aggregate": aggregate_comparisons(rows),
            }
            output = args.output_dir / f"{dataset}_{mode}.json"
            save_comparison(output, payload)
            table_rows.append(_flatten(dataset, mode, payload))

    if not table_rows:
        raise RuntimeError("no final_graph.json files were found")
    columns = list(table_rows[0])
    with (args.output_dir / "extended_structure.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(table_rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
