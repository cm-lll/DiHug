#!/usr/bin/env python3
"""
IMDB 预处理：node.dat + IMDB_movie.csv + movie_metadata.csv
 -> nodes.jsonl + edges.jsonl（含子类别）

只保留直接关系：导演指导电影、演员参演电影。去掉 0-1、1-1 间接关系。
题材 (Genre) 仅用于给电影赋子类别，不作为图中节点。

节点：Director, Actor, Movie（无 Genre）
边：director_movie（导演→电影）, actor_movie（演员→电影）

子类别:
  - Movie: 题材（genre）作为子类别
  - Director/Actor: 按作品数量 High/Medium/Low
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

TYPE_NAMES = {0: "Director", 1: "Actor", 2: "Movie", 3: "Genre"}
# 只保留直接关系：0-2 导演→电影, 1-2 演员→电影；2-3 仅用于取电影题材，不输出
EDGE_TYPE_MAP = {
    "0-2": ("Director", "Movie", "director_movie"),
    "1-2": ("Actor", "Movie", "actor_movie"),
    "2-3": ("Movie", "Genre", "movie_genre"),  # 仅用于赋子类别，边不输出
}
# 输出到图中的边类型
OUTPUT_EDGE_TYPES = {"director_movie", "actor_movie"}
GENRE_SCHEMA = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
    "Music", "Mystery", "Romance", "Sci-Fi", "Thriller", "Western", "Other",
]


def normalize_name(s: str) -> str:
    """标准化名称便于匹配：转小写、空格/特殊字符统一"""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s


def load_nodes(node_dat: Path) -> list:
    """加载 node.dat: id, type, name"""
    nodes = []
    with open(node_dat, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nid = int(parts[0])
                ntype = int(parts[-1])
                name = " ".join(parts[1:-1]).replace(" ", " ") if len(parts) > 2 else ""
                nodes.append({"id": nid, "type": TYPE_NAMES[ntype], "name": name})
    return nodes


def load_edges(csv_path: Path) -> list:
    """加载 IMDB_movie.csv 边表。注意：CSV 中 0-2/1-2 的 source 实际为电影、dest 为人，需交换为 人→电影"""
    df = pd.read_csv(csv_path)
    edges = []
    for _, row in df.iterrows():
        edge_class = str(row["edge_class"])
        if edge_class not in EDGE_TYPE_MAP:
            continue
        src_type, dst_type, etype = EDGE_TYPE_MAP[edge_class]
        s, d = int(row["source_node"]), int(row["dest_node"])
        # 0-2/1-2: CSV 中 source=电影 dest=人，需交换为人→电影
        if etype in OUTPUT_EDGE_TYPES:
            s, d = d, s
        edges.append({"src": s, "dst": d, "type": etype})
    return edges


def assign_movie_genres(nodes: list, edges: list, metadata_path: Path) -> dict:
    """
    为电影分配子类别（主 genre）：
    1. 优先从 Movie-Genre 边获取
    2. 若无边，用 movie_metadata 的 genres 匹配（按名）
    """
    id2node = {n["id"]: n for n in nodes}
    movie_ids = [n["id"] for n in nodes if n["type"] == "Movie"]
    genre_ids = [n["id"] for n in nodes if n["type"] == "Genre"]

    # Movie -> Genre 边（可能多条，取第一个或多数）
    movie_to_genres = defaultdict(list)
    for e in edges:
        if e["type"] == "movie_genre":
            mid, gid = e["src"], e["dst"]
            gname = id2node.get(gid, {}).get("name", "")
            if gname:
                movie_to_genres[mid].append(gname)

    movie_subtype = {}
    for mid in movie_ids:
        genres = movie_to_genres.get(mid, [])
        if genres:
            primary = genres[0]
            if primary not in GENRE_SCHEMA:
                primary = "Other"
            movie_subtype[mid] = primary
        else:
            movie_subtype[mid] = "Other"

    # 用 movie_metadata 补充缺失
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        node_names = {n["id"]: normalize_name(n["name"]) for n in nodes if n["type"] == "Movie"}
        meta["norm_title"] = meta["movie_title"].fillna("").apply(normalize_name)

        for n in nodes:
            if n["type"] != "Movie":
                continue
            mid, name = n["id"], n["name"]
            if movie_subtype.get(mid) != "Other" and movie_to_genres.get(mid):
                continue
            norm = normalize_name(name.replace("-", " "))
            match = meta[meta["norm_title"] == norm]
            if len(match) > 0 and pd.notna(match.iloc[0].get("genres")):
                g = str(match.iloc[0]["genres"]).split("|")[0]
                if g in GENRE_SCHEMA:
                    movie_subtype[mid] = g
                else:
                    movie_subtype[mid] = "Other"

    return movie_subtype


def assign_person_subtype_by_count(nodes: list, edges: list) -> dict:
    """Director/Actor 子类别 = 按作品数量 High/Medium/Low"""
    person_count = defaultdict(int)
    for e in edges:
        if e["type"] == "actor_movie":
            person_count[e["src"]] += 1
        elif e["type"] == "director_movie":
            person_count[e["src"]] += 1

    # 固定阈值：High=4+, Medium=2-3, Low=1（因数据中大多数人为1-2部作品）
    person_subtype = {}
    for node in nodes:
        if node["type"] not in ("Actor", "Director"):
            continue
        pid = node["id"]
        c = person_count.get(pid, 0)
        if c >= 4:
            person_subtype[pid] = "High"
        elif c >= 2:
            person_subtype[pid] = "Medium"
        else:
            person_subtype[pid] = "Low"

    return person_subtype


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/raw/IMDB_movie")
    parser.add_argument("--output-dir", type=str, default="data/IMDB_processed")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载 node.dat ...")
    nodes = load_nodes(in_dir / "node.dat")
    print(f"      共 {len(nodes)} 节点")

    print("[2/4] 加载边表 ...")
    edges = load_edges(in_dir / "IMDB_movie.csv")
    print(f"      共 {len(edges)} 条边")

    print("[3/4] 划分子类别 ...")
    movie_subtype = assign_movie_genres(nodes, edges, in_dir / "movie_metadata.csv")
    person_subtype = assign_person_subtype_by_count(nodes, edges)

    for n in nodes:
        nid = n["id"]
        t = n["type"]
        if t == "Movie":
            n["subtype"] = movie_subtype.get(nid, "Other")
        elif t in ("Actor", "Director"):
            n["subtype"] = person_subtype.get(nid, "Other")
        else:
            n["subtype"] = "Other"

    # 只输出 Director, Actor, Movie 节点；只输出 director_movie, actor_movie 边
    out_nodes = [n for n in nodes if n["type"] in ("Director", "Actor", "Movie")]
    out_edges = [e for e in edges if e["type"] in OUTPUT_EDGE_TYPES]

    print("[4/4] 写入 nodes.jsonl, edges.jsonl ...")

    with open(out_dir / "nodes.jsonl", "w", encoding="utf-8") as f:
        for n in out_nodes:
            f.write(json.dumps({"id": str(n["id"]), "type": n["type"], "subtype": n["subtype"]}, ensure_ascii=False) + "\n")

    with open(out_dir / "edges.jsonl", "w", encoding="utf-8") as f:
        for e in out_edges:
            f.write(json.dumps({"src": str(e["src"]), "dst": str(e["dst"]), "type": e["type"]}, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    movie_cnt = Counter(n["subtype"] for n in out_nodes if n["type"] == "Movie")
    person_cnt = Counter(n["subtype"] for n in out_nodes if n["type"] in ("Director", "Actor"))
    print(f"\n[Movie 子类别分布] {dict(movie_cnt.most_common(10))}")
    print(f"[Director/Actor 子类别分布] {dict(person_cnt)}")
    print(f"[DONE] 输出: {out_dir}/nodes.jsonl, edges.jsonl")


if __name__ == "__main__":
    main()
