#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from collections import defaultdict, Counter
import torch

# ---------- 工具函数 ----------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_subtype(x):
    if x is None:
        return "Other"
    if isinstance(x, list) and len(x) > 0:
        return str(x[0])
    if isinstance(x, str):
        return x
    return str(x)

# ---------- Step 1: 清洗 edges ----------
def clean_edges(in_path: Path, out_path: Path):
    aff_by_author = defaultdict(list)
    ao_by_paper_role = defaultdict(lambda: defaultdict(list))
    ao_all_pairs = defaultdict(set)
    cites_set = set()

    for e in read_jsonl(in_path):
        t = e.get("type")
        if t == "affiliated_with":
            aff_by_author[e["src"]].append(e["dst"])
        elif t == "author_of":
            a, p = e["src"], e["dst"]
            role = e.get("subtype", "co_author")
            ao_by_paper_role[p][role].append(a)
            ao_all_pairs[p].add(a)
        elif t == "cites":
            s, d = e["src"], e["dst"]
            cites_set.add((s, d))

    out = []
    # 1) affiliated_with
    for a, orgs in aff_by_author.items():
        if orgs:
            c = Counter(orgs)
            primary_org, _ = c.most_common(1)[0]
            out.append({"src": a, "dst": primary_org, "type": "affiliated_with"})
    # 2) author_of
    for p, role_map in ao_by_paper_role.items():
        first = None
        second = None
        if "first_author" in role_map and role_map["first_author"]:
            first = Counter(role_map["first_author"]).most_common(1)[0][0]
        if "second_author" in role_map and role_map["second_author"]:
            cand = Counter(role_map["second_author"])
            if first and first in cand and len(cand) > 1:
                cand[first] = -10**9
            second = cand.most_common(1)[0][0]
        all_authors = set(ao_all_pairs.get(p, set()))
        for rr, lst in role_map.items():
            all_authors.update(lst)
        if first:
            out.append({"src": first, "dst": p, "type": "author_of", "subtype": "first_author"})
        if second and second != first:
            out.append({"src": second, "dst": p, "type": "author_of", "subtype": "second_author"})
        for a in sorted(all_authors):
            if a not in (first, second):
                out.append({"src": a, "dst": p, "type": "author_of", "subtype": "co_author"})
    # 3) cites
    for s, d in sorted(cites_set):
        if s != d:
            out.append({"src": s, "dst": d, "type": "cites"})
    # 全局去重
    seen = set()
    deduped = []
    for e in out:
        key = (e.get("src"), e.get("dst"), e.get("type"), e.get("subtype", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    write_jsonl(out_path, deduped)
    print("[cleaned]", out_path, "counts:", Counter([e["type"] for e in deduped]))
    return out_path

# ---------- Step 2: 导出 .pt + 生成 meta.json ----------
def export_pt(subgraph_dir: Path, schema_path: Path, edges_path: Path, out_dir: Path):
    # 加载 schema（每个节点类型的子类型列表）
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    node_types = sorted(schema.keys())

    # 子类型名称到ID
    subtype2id = {nt: {name: i for i, name in enumerate(schema[nt])} for nt in node_types}

    # 读取节点，建立全局ID -> (类型, 本地索引)
    gid2info = {}
    per_type_ids = {nt: [] for nt in node_types}
    per_type_sub = {nt: [] for nt in node_types}

    typed_nodes = subgraph_dir / "typed_nodes"
    for nt in node_types:
        p = typed_nodes / f"{nt}.jsonl"
        if not p.exists():
            continue
        for r in read_jsonl(p):
            nid = str(r["id"])
            st_name = normalize_subtype(r.get("subtype"))
            if st_name not in subtype2id[nt]:
                st_name = "Other" if "Other" in subtype2id[nt] else next(iter(subtype2id[nt]))
            st_id = subtype2id[nt][st_name]
            if nid in gid2info:
                continue
            gid2info[nid] = (nt, len(per_type_ids[nt]))
            per_type_ids[nt].append(nid)
            per_type_sub[nt].append(st_id)

    # 读取边，构造每个家族的张量与标签词表
    fam2edges = defaultdict(list)
    fam_vocab = {}  # fam -> {label_name: id}
    unknown_edges_dropped = 0

    for e in read_jsonl(edges_path):
        s, t = str(e["src"]), str(e["dst"])
        if s not in gid2info or t not in gid2info:
            unknown_edges_dropped += 1
            continue
        fam = str(e["type"])
        lname = f"{fam}:{e.get('subtype','__none__')}"
        fam2edges[fam].append((s, t, lname))

    fam_data = {}
    for fam, elist in fam2edges.items():
        vocab = {}
        src_local = []
        dst_local = []
        y = []
        for s, t, lname in elist:
            st, si = gid2info[s]
            tt, ti = gid2info[t]
            if lname not in vocab:
                vocab[lname] = len(vocab) + 1  # 标签ID从1开始
            src_local.append(si)
            dst_local.append(ti)
            y.append(vocab[lname])
        fam_data[fam] = {
            "src_local": torch.tensor(src_local, dtype=torch.long),
            "dst_local": torch.tensor(dst_local, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
        }
        fam_vocab[fam] = vocab

    # 保存 .pt
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_t = {
        nt: {
            "ids": per_type_ids[nt],
            "subtype": torch.tensor(per_type_sub[nt], dtype=torch.long),
            "A": len(subtype2id[nt]),
        }
        for nt in node_types
        if per_type_ids[nt]
    }
    torch.save(nodes_t, out_dir / "nodes.pt")
    torch.save({"families": fam_data}, out_dir / "edges.pt")
    print("[exported]", out_dir)

    # 生成 meta.json
    fam_endpoints = {
        "affiliated_with": {"src_type": "Author", "dst_type": "Organization"},
        "author_of": {"src_type": "Author", "dst_type": "Paper"},
        "cites": {"src_type": "Paper", "dst_type": "Paper"},
    }

    fam_label2id = {fam: {label: idx for label, idx in vocab.items()} for fam, vocab in fam_vocab.items()}
    fam_id2label = {
        fam: {str(idx): label for label, idx in vocab.items()}
        for fam, vocab in fam_vocab.items()
    }

    meta = {
        "node_types": node_types,
        "schema_by_type": schema,
        "fam_endpoints": fam_endpoints,
        "fam_label2id": fam_label2id,
        "fam_id2label": fam_id2label,
        "unknown_edges_dropped": unknown_edges_dropped,
        "num_nodes_by_type": {nt: len(per_type_ids[nt]) for nt in node_types},
        "num_edges_by_family": {fam: int(fam_data[fam]["y"].numel()) for fam in fam_data.keys()},
    }

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[meta saved]", out_dir / "meta.json")

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph_dir", required=True)
    ap.add_argument("--schema_by_type", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    sg = Path(args.subgraph_dir)
    out = Path(args.out_dir)

    # Step1: 清洗
    clean_path = clean_edges(sg / "edges.jsonl", sg / "edges_clean.jsonl")

    # Step2: 导出 + 生成 meta.json
    export_pt(sg, Path(args.schema_by_type), clean_path, out)

if __name__ == "__main__":
    main()
