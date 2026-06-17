#!/usr/bin/env python3
"""
PGB (PubMed Graph Benchmark) 预处理：异质图
 -> nodes.jsonl + edges.jsonl（含子类别）

节点：Paper, Author
边：author_paper（作者→论文）, paper_paper（论文→论文，引用）

子类别:
  - Paper: 从 publication_type 取首标签（Review/Clinical Trial/...），无则 Other
  - Author: 按发表论文数量 High(4+)/Medium(2-3)/Low(1)
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

# publication_type 常见值 -> 子类别（与 extract 的 schema 一致）
PUBTYPE_TO_SUBTYPE = {
    "Review": "Review",
    "Clinical Trial": "Clinical",
    "Randomized Controlled Trial": "Clinical",
    "Journal Article": "Article",
    "Clinical Trial, Phase I": "Clinical",
    "Clinical Trial, Phase II": "Clinical",
    "Clinical Trial, Phase III": "Clinical",
    "Case Reports": "CaseReport",
    "Meta-Analysis": "Review",
    "Comparative Study": "Study",
    "Evaluation Study": "Study",
    "Observational Study": "Study",
    "Multicenter Study": "Study",
    "Letter": "Other",
    "Editorial": "Other",
}


def author_key(a: dict) -> str:
    """生成作者唯一标识（用于去重）"""
    first = a.get("first") or ""
    last = a.get("last") or ""
    mid = a.get("middle") or []
    mid_str = "-".join(mid) if isinstance(mid, list) else str(mid)
    raw = f"{first}|{mid_str}|{last}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()[:16] if raw else ""


def paper_subtype(rec: dict) -> str:
    """从 publication_type 取论文子类别"""
    pt = rec.get("publication_type") or []
    if isinstance(pt, str):
        pt = [pt]
    if not pt:
        return "Other"
    first = pt[0] if isinstance(pt[0], str) else str(pt[0])
    return PUBTYPE_TO_SUBTYPE.get(first, "Other")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/raw/PGB/extracted",
                        help="PGB 解压目录，或含 jsonl 的目录")
    parser.add_argument("--input-file", type=str, default=None,
                        help="或直接指定单个 jsonl 文件")
    parser.add_argument("--output-dir", type=str, default="data/PubMed_PGB_processed")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少篇论文，0=全部")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 jsonl 路径
    jsonl_paths = []
    if args.input_file:
        p = Path(args.input_file)
        if p.exists():
            jsonl_paths = [p]
    else:
        for p in in_dir.rglob("*.jsonl"):
            jsonl_paths.append(p)
        if not jsonl_paths:
            for p in in_dir.rglob("*.json"):
                if p.suffix == ".json":
                    jsonl_paths.append(p)
        jsonl_paths.sort()

    if not jsonl_paths:
        raise FileNotFoundError(
            f"未找到 jsonl 文件。请先运行: python dataprocess/scripts/download_pgb.py "
            f"--output-dir data/raw/PGB\n然后解压到 {in_dir}"
        )

    print(f"[1/4] 发现 {len(jsonl_paths)} 个 jsonl 文件")

    pmid2paper = {}
    author2papers = defaultdict(set)  # author_id -> set(pmid)
    paper_paper_edges = []  # (src_pmid, dst_pmid)
    n_read = 0

    print("[2/4] 读取论文与作者...")
    for fp in jsonl_paths:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pmid = rec.get("pmid") or rec.get("pm_id")
                if not pmid:
                    continue
                pmid = str(pmid)
                n_read += 1
                if args.limit > 0 and n_read > args.limit:
                    break

                subtype = paper_subtype(rec)
                pmid2paper[pmid] = {"id": pmid, "type": "Paper", "subtype": subtype}

                # author_paper
                authors = rec.get("authors") or []
                for a in authors:
                    aid = author_key(a)
                    if aid:
                        author2papers[aid].add(pmid)

                # paper_paper
                out_cites = rec.get("outbound_citations") or rec.get("outbound_citation") or []
                if isinstance(out_cites, str):
                    out_cites = [out_cites] if out_cites else []
                for c in out_cites:
                    c = str(c)
                    if c in pmid2paper or c:  # 目标论文可能在后续出现
                        paper_paper_edges.append((pmid, c))

        if args.limit > 0 and n_read >= args.limit:
            break

    # 只保留两边都在的 paper_paper 边
    pmids = set(pmid2paper.keys())
    paper_paper_edges = [(s, d) for s, d in paper_paper_edges if s in pmids and d in pmids]

    # 只保留至少发 1 篇论文且论文在 pmids 内的作者
    author2papers = {aid: p & pmids for aid, p in author2papers.items() if p & pmids}

    print(f"      论文: {len(pmid2paper)}, 作者: {len(author2papers)}, paper_paper边: {len(paper_paper_edges)}")

    print("[3/4] 划分子类别...")
    # Author 子类别: 按论文数
    author_subtype = {}
    for aid, papers in author2papers.items():
        c = len(papers)
        if c >= 4:
            author_subtype[aid] = "High"
        elif c >= 2:
            author_subtype[aid] = "Medium"
        else:
            author_subtype[aid] = "Low"

    nodes = []
    for pmid, info in pmid2paper.items():
        nodes.append(info)
    for aid in author2papers:
        nodes.append({"id": f"author_{aid}", "type": "Author", "subtype": author_subtype.get(aid, "Low")})

    edges = []
    for aid, papers in author2papers.items():
        for pmid in papers:
            edges.append({"src": f"author_{aid}", "dst": pmid, "type": "author_paper"})
    for s, d in paper_paper_edges:
        edges.append({"src": s, "dst": d, "type": "paper_paper"})

    print("[4/4] 写入 nodes.jsonl, edges.jsonl ...")
    with open(out_dir / "nodes.jsonl", "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    with open(out_dir / "edges.jsonl", "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    pcnt = Counter(n["subtype"] for n in nodes if n["type"] == "Paper")
    acnt = Counter(n["subtype"] for n in nodes if n["type"] == "Author")
    print(f"\n[Paper 子类别分布] {dict(pcnt.most_common(10))}")
    print(f"[Author 子类别分布] {dict(acnt)}")
    print(f"[DONE] 输出: {out_dir}/nodes.jsonl, edges.jsonl")


if __name__ == "__main__":
    main()
