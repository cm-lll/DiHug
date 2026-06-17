#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph_dir", type=str, required=True,
                    help="e.g. preprocess/output/ACM_subgraphs/subgraph_000")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    subgraph_dir = Path(args.subgraph_dir)
    edges_path = subgraph_dir / "edges.jsonl"
    assert edges_path.exists(), f"not found: {edges_path}"

    # ---- collect node ids per type (optional but helps detect missing affiliations) ----
    typed_nodes_dir = subgraph_dir / "typed_nodes"
    authors = set()
    orgs = set()
    papers = set()

    def load_ids(jsonl_path: Path):
        s = set()
        if not jsonl_path.exists():
            return s
        for obj in read_jsonl(jsonl_path):
            if "id" in obj:
                s.add(obj["id"])
        return s

    authors = load_ids(typed_nodes_dir / "Author.jsonl")
    orgs = load_ids(typed_nodes_dir / "Organization.jsonl")
    papers = load_ids(typed_nodes_dir / "Paper.jsonl")

    # ---- affiliation stats ----
    aff_dst_by_author = defaultdict(set)     # author -> {org_id}
    aff_edges_by_author = defaultdict(list)  # author -> [org_id in edges order] (keep duplicates if any)
    aff_total = 0

    # ---- role uniqueness checks for author_of ----
    # paper -> Counter(role)
    role_count_by_paper = defaultdict(Counter)
    author_of_total = 0

    # ---- cites sanity ----
    cites_total = 0
    cites_self_loops = 0
    cites_pairs = set()
    cites_bidir = 0

    for e in read_jsonl(edges_path):
        et = e.get("type")
        if et == "affiliated_with":
            a = e.get("src")
            o = e.get("dst")
            aff_total += 1
            aff_dst_by_author[a].add(o)
            aff_edges_by_author[a].append(o)
        elif et == "author_of":
            author_of_total += 1
            a = e.get("src")
            p = e.get("dst")
            role = e.get("subtype", "__none__")
            role_count_by_paper[p][role] += 1
        elif et == "cites":
            cites_total += 1
            s = e.get("src")
            d = e.get("dst")
            if s == d:
                cites_self_loops += 1
            # bidirectional count (undirected pair)
            key = (s, d)
            rkey = (d, s)
            if rkey in cites_pairs:
                cites_bidir += 1
            cites_pairs.add(key)

    # ---------- report: affiliated_with ----------
    # per-author number of unique orgs
    uniq_counts = Counter({a: len(orgset) for a, orgset in aff_dst_by_author.items()})
    # distribution
    dist = Counter(uniq_counts.values())

    num_authors = len(authors) if authors else len(set(list(aff_dst_by_author.keys())))
    num_with_aff = len(aff_dst_by_author)
    num_multi = sum(1 for a, k in uniq_counts.items() if k > 1)
    num_single = sum(1 for a, k in uniq_counts.items() if k == 1)
    num_zero = None
    if authors:
        num_zero = len(authors) - num_with_aff

    print("=== affiliated_with multiplicity check ===")
    print(f"edges.affiliated_with total edges: {aff_total}")
    print(f"authors in typed_nodes: {len(authors) if authors else '(unknown)'}")
    print(f"authors with >=1 affiliated_with: {num_with_aff}")
    if num_zero is not None:
        print(f"authors with 0 affiliated_with: {num_zero}")
    print(f"authors with exactly 1 org: {num_single}")
    print(f"authors with >1 org (multi-affiliation): {num_multi}")
    print("unique-org count distribution (k -> #authors):")
    for k in sorted(dist.keys()):
        print(f"  k={k}: {dist[k]}")

    # show examples of multi-affiliation
    if num_multi > 0:
        # sort by unique org count descending
        multi_sorted = sorted(
            [(a, len(aff_dst_by_author[a])) for a in aff_dst_by_author.keys() if len(aff_dst_by_author[a]) > 1],
            key=lambda x: (-x[1], x[0])
        )
        print(f"\nTop {min(args.topk, len(multi_sorted))} multi-affiliation authors:")
        for a, k in multi_sorted[: args.topk]:
            org_list = sorted(list(aff_dst_by_author[a]))
            # also show if there were duplicate edges
            dup_edges = len(aff_edges_by_author[a]) - len(set(aff_edges_by_author[a]))
            print(f"- author={a} unique_orgs={k} total_edges={len(aff_edges_by_author[a])} dup_edges={dup_edges}")
            print(f"  orgs={org_list}")

    # ---------- report: author_of role uniqueness ----------
    # For each paper, check if multiple first_author / second_author
    multi_first = []
    multi_second = []
    for p, cc in role_count_by_paper.items():
        if cc.get("first_author", 0) > 1:
            multi_first.append((p, cc["first_author"]))
        if cc.get("second_author", 0) > 1:
            multi_second.append((p, cc["second_author"]))

    print("\n=== author_of role sanity (per paper) ===")
    print(f"edges.author_of total edges: {author_of_total}")
    print(f"papers in typed_nodes: {len(papers) if papers else '(unknown)'}")
    print(f"papers with >1 first_author: {len(multi_first)}")
    print(f"papers with >1 second_author: {len(multi_second)}")
    if multi_first:
        print(f"  examples first_author duplicates: {multi_first[: min(args.topk, len(multi_first))]}")
    if multi_second:
        print(f"  examples second_author duplicates: {multi_second[: min(args.topk, len(multi_second))]}")

    # ---------- report: cites ----------
    print("\n=== cites sanity ===")
    print(f"edges.cites total edges: {cites_total}")
    print(f"self_loops: {cites_self_loops}")
    print(f"bidirectional pairs (counted as occurrences where reverse already seen): {cites_bidir}")


if __name__ == "__main__":
    main()
