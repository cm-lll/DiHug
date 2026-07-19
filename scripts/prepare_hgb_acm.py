#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch


TYPE_NAMES = ("Paper", "Author", "Subject", "Term")


def _selected_papers(payload: dict, seed: int) -> np.ndarray:
    paper_conf = payload["PvsC"].nonzero()[1]
    database = np.flatnonzero(np.isin(paper_conf, [1, 13]))
    data_mining = np.flatnonzero(np.isin(paper_conf, [0]))
    wireless = np.flatnonzero(np.isin(paper_conf, [9, 10]))
    rng = np.random.RandomState(int(seed))
    sampled_database = np.sort(rng.choice(database, 994, replace=False))
    return np.sort(np.concatenate((sampled_database, data_mining, wireless)))


def _unique_indices(matrix, selected_rows: np.ndarray) -> np.ndarray:
    return np.unique(matrix[selected_rows].nonzero()[1]).astype(np.int64)


def _bipartite_edges(matrix, rows: np.ndarray, columns: np.ndarray):
    column_to_local = {int(value): index for index, value in enumerate(columns.tolist())}
    local_rows, raw_columns = matrix[rows][:, columns].nonzero()
    # scipy has already remapped the sliced columns, so raw_columns are local.
    del column_to_local
    return sorted({(int(left), int(right)) for left, right in zip(local_rows, raw_columns)})


def _citation_edges(matrix, papers: np.ndarray):
    left, right = matrix[papers][:, papers].nonzero()
    return sorted(
        {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in zip(left, right)
            if int(u) != int(v)
        }
    )


def _family_payload(edges):
    return {
        "src_local": torch.tensor([edge[0] for edge in edges], dtype=torch.long),
        "dst_local": torch.tensor([edge[1] for edge in edges], dtype=torch.long),
        "y": torch.ones(len(edges), dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official HGB ACM.mat to DiHug.")
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    digest = hashlib.sha256(args.mat.read_bytes()).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise RuntimeError(
            f"ACM.mat SHA256 mismatch: expected {args.expected_sha256}, found {digest}"
        )
    payload = scipy.io.loadmat(args.mat)
    papers = _selected_papers(payload, args.seed)
    authors = _unique_indices(payload["PvsA"], papers)
    subjects = _unique_indices(payload["PvsL"], papers)
    terms = _unique_indices(payload["TvsP"].transpose(), papers)
    raw_by_type = {
        "Paper": papers,
        "Author": authors,
        "Subject": subjects,
        "Term": terms,
    }
    offsets = {}
    offset = 0
    nodes = {}
    for type_name in TYPE_NAMES:
        count = int(len(raw_by_type[type_name]))
        offsets[type_name] = offset
        nodes[type_name] = {
            "ids": list(range(offset, offset + count)),
            "subtype": torch.zeros(count, dtype=torch.long),
            "A": 1,
        }
        offset += count

    families = {
        "link_Paper_Paper": _citation_edges(payload["PvsP"], papers),
        "link_Paper_Author": _bipartite_edges(payload["PvsA"], papers, authors),
        "link_Paper_Subject": _bipartite_edges(payload["PvsL"], papers, subjects),
        "link_Paper_Term": _bipartite_edges(
            payload["TvsP"].transpose(), papers, terms
        ),
    }
    endpoints = {
        "link_Paper_Paper": ("Paper", "Paper"),
        "link_Paper_Author": ("Paper", "Author"),
        "link_Paper_Subject": ("Paper", "Subject"),
        "link_Paper_Term": ("Paper", "Term"),
    }
    edges = {"families": {name: _family_payload(value) for name, value in families.items()}}
    meta = {
        "node_types": list(TYPE_NAMES),
        "schema_by_type": {name: ["default"] for name in TYPE_NAMES},
        "fam_endpoints": {
            name: {"src_type": endpoint[0], "dst_type": endpoint[1]}
            for name, endpoint in endpoints.items()
        },
        "fam_label2id": {name: {f"{name}:__none__": 1} for name in families},
        "fam_id2label": {name: {"1": f"{name}:__none__"} for name in families},
        "num_nodes_by_type": {name: len(raw_by_type[name]) for name in TYPE_NAMES},
        "num_edges_by_family": {name: len(value) for name, value in families.items()},
        "num_undirected_edges": int(sum(len(value) for value in families.values())),
        "undirected_ground_truth": True,
        "bidirectional_mode": "collapse_citation_reverse_and_canonical_single",
        "source": "HGB ACM derived with official GTN seed-2020 paper selection",
        "source_sha256": digest,
        "preprocessing": {
            "paper_selection_seed": int(args.seed),
            "database_papers_sampled": 994,
            "paper_citation_direction": "collapsed_to_undirected_union",
            "original_relation_families": [
                "paper-cite-paper",
                "paper-ref-paper",
                "paper-author",
                "paper-subject",
                "paper-term",
            ],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(nodes, args.output / "nodes.pt")
    torch.save(edges, args.output / "edges.pt")
    (args.output / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
