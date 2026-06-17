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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="edges_clean.jsonl")
    args = ap.parse_args()

    subgraph_dir = Path(args.subgraph_dir)
    in_path = subgraph_dir / "edges.jsonl"
    out_path = subgraph_dir / args.out_name
    assert in_path.exists(), f"not found: {in_path}"

    # -------- collect raw edges --------
    aff_by_author = defaultdict(list)  # author -> [org]
    ao_by_paper_role = defaultdict(lambda: defaultdict(list))  # paper -> role -> [author]
    ao_all_pairs = defaultdict(set)  # paper -> set(author) to keep coverage
    cites_set = set()

    for e in read_jsonl(in_path):
        t = e.get("type")
        if t == "affiliated_with":
            aff_by_author[e["src"]].append(e["dst"])
        elif t == "author_of":
            a = e["src"]
            p = e["dst"]
            role = e.get("subtype", "co_author")
            ao_by_paper_role[p][role].append(a)
            ao_all_pairs[p].add(a)
        elif t == "cites":
            s, d = e["src"], e["dst"]
            cites_set.add((s, d))

    # -------- build cleaned edges --------
    out = []

    # 1) affiliated_with: keep ONE primary org per author (most frequent)
    for a, orgs in aff_by_author.items():
        if not orgs:
            continue
        c = Counter(orgs)
        primary_org, _cnt = c.most_common(1)[0]
        out.append({"src": a, "dst": primary_org, "type": "affiliated_with"})

    # 2) author_of: enforce unique first/second per paper, others co_author
    for p, role_map in ao_by_paper_role.items():
        # pick first_author
        first = None
        if "first_author" in role_map and role_map["first_author"]:
            first = Counter(role_map["first_author"]).most_common(1)[0][0]

        second = None
        if "second_author" in role_map and role_map["second_author"]:
            cand = Counter(role_map["second_author"])
            # avoid selecting same as first if possible
            if first is not None and first in cand and len(cand) > 1:
                cand[first] = -10**9
            second = cand.most_common(1)[0][0]

        # all authors connected to this paper
        all_authors = set()
        for a in ao_all_pairs.get(p, set()):
            all_authors.add(a)
        # also include any role-mapped authors
        for rr, lst in role_map.items():
            all_authors.update(lst)

        # emit edges
        if first is not None:
            out.append({"src": first, "dst": p, "type": "author_of", "subtype": "first_author"})
        if second is not None and second != first:
            out.append({"src": second, "dst": p, "type": "author_of", "subtype": "second_author"})

        for a in sorted(all_authors):
            if a == first or a == second:
                continue
            out.append({"src": a, "dst": p, "type": "author_of", "subtype": "co_author"})

    # 3) cites: dedup
    for s, d in sorted(cites_set):
        if s == d:
            continue
        out.append({"src": s, "dst": d, "type": "cites"})

    # final global dedup (just in case)
    seen = set()
    deduped = []
    for e in out:
        key = (e.get("src"), e.get("dst"), e.get("type"), e.get("subtype", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    write_jsonl(out_path, deduped)

    # -------- print summary --------
    ctype = Counter([e["type"] for e in deduped])
    print("[saved]", out_path)
    print("counts:", dict(ctype))


if __name__ == "__main__":
    main()
