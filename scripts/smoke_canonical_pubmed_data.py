#!/usr/bin/env python3
"""Verify PubMed baseline uses canonical single directed edges (not link+rev duplicate)."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "sparse_diffusion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from sparse_diffusion.metrics.sampling_metrics import _batch_to_nx_graphs


def main() -> None:
    proc = ROOT / "data/PubMed_baseline_subgraphs/processed/train.pt"
    meta_path = ROOT / "data/PubMed_baseline_subgraphs/subgraph_000/meta.json"
    if not proc.exists():
        raise SystemExit(f"Missing {proc}; run convert_hgen_pubmed_baseline.py first.")

    data = torch.load(proc, weights_only=False)
    if isinstance(data, tuple):
        data = data[0]

    ei = data.edge_index
    directed = ei.size(1)
    arcs = set(map(tuple, ei.T.tolist()))
    und = set()
    for u, v in arcs:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        und.add((a, b))
    both = sum(1 for a, b in und if (a, b) in arcs and (b, a) in arcs)

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mode = meta.get("bidirectional_mode", "?")
    n_fam = len(meta.get("fam_endpoints", {}))
    gt_und = meta.get("num_undirected_edges", "?")

    gs = _batch_to_nx_graphs(data)
    nx_edges = gs[0].number_of_edges()

    print(f"bidirectional_mode={mode}")
    print(f"num_families={n_fam}")
    print(f"directed_edges={directed}")
    print(f"unique_undirected_pairs={len(und)}")
    print(f"pairs_with_both_directions={both}")
    print(f"nx_eval_edges={nx_edges}")
    print(f"meta num_undirected_edges={gt_und}")

    ok = (
        mode == "canonical_single"
        and n_fam == 16
        and directed == len(und)
        and both == 0
        and nx_edges == len(und)
        and str(gt_und) == str(len(und))
    )
    if not ok:
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
