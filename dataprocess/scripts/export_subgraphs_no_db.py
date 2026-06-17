# scripts/export_subgraphs_full_jsonl_no_db.py
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Set, List, Tuple, Optional


# -------- robust JSON reader (handles '}{' concatenated JSON in one line) --------
def iter_json_objects_from_text(text: str) -> Iterator[Dict[str, Any]]:
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                yield obj
            i = j
        except json.JSONDecodeError:
            break

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            for obj in iter_json_objects_from_text(s):
                yield obj

def is_paper(obj: Dict[str, Any]) -> bool:
    # 只认顶层 paper（与之前一致）：必须有 title/year/references(list)
    pid = obj.get("id")
    title = obj.get("title")
    year = obj.get("year")
    refs = obj.get("references", None)
    if not isinstance(pid, str) or not pid:
        return False
    if not isinstance(title, str) or not title.strip():
        return False
    if not isinstance(year, int) or year < 1800 or year > 2030:
        return False
    if not isinstance(refs, list):
        return False
    return True


# -------- Pass 1: collect citers for each seed (1-hop) --------
def collect_citers(
    in_path: Path,
    seeds: List[str],
) -> Dict[str, Set[str]]:
    seed_set = set(seeds)
    citers: Dict[str, Set[str]] = {s: set() for s in seeds}

    seen = 0
    t0 = time.time()
    for obj in iter_jsonl(in_path):
        seen += 1
        if not is_paper(obj):
            continue
        src = obj["id"]
        refs = obj.get("references") or []
        for r in refs:
            if isinstance(r, str) and r in seed_set:
                citers[r].add(src)

        if seen % 500_000 == 0:
            elapsed = time.time() - t0
            sample = [len(citers[s]) for s in seeds[:5]]
            print(f"[PASS1] seen_objs={seen:,} time={elapsed:.1f}s sample_citers_sizes(first5)={sample}")

    return citers


def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/export_subgraphs_full_jsonl_no_db.py <input_jsonl> <seeds_100.json> <out_dir> [min_nodes] [max_nodes]")
        print(r"Example: python scripts\export_subgraphs_full_jsonl_no_db.py .\datasets\ACM-Citation-network-V12.jsonl .\datasets\seeds_100.json .\datasets\ACM_subgraphs_full 100 500")
        sys.exit(1)

    in_path = Path(sys.argv[1]).resolve()
    seeds_path = Path(sys.argv[2]).resolve()
    out_dir = Path(sys.argv[3]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    min_nodes = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    max_nodes = int(sys.argv[5]) if len(sys.argv) >= 6 else 500

    seeds_obj = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds = [x["id"] for x in seeds_obj.get("selected", [])]
    if not seeds:
        raise RuntimeError("No seeds found in seeds file.")

    print(f"[INFO] seeds={len(seeds)} node_range=[{min_nodes},{max_nodes}]")
    print("[INFO] PASS 1/2: collect 1-hop citers for each seed...")

    # PASS 1: collect citers sets
    citers = collect_citers(in_path, seeds)

    # Decide which seeds are valid, build node-need sets
    valid_seeds: List[Tuple[str, Set[str]]] = []
    for s in seeds:
        V = set(citers[s])
        V.add(s)
        if min_nodes <= len(V) <= max_nodes:
            valid_seeds.append((s, V))

    print(f"[INFO] PASS 1 done. valid_subgraphs={len(valid_seeds)} / {len(seeds)}")
    if not valid_seeds:
        print("[WARN] No valid subgraphs in this node range. Adjust min/max or re-pick seeds.")
        return

    # Map paper_id -> list of subgraph indices that need it
    need_map: Dict[str, List[int]] = {}
    for idx, (_seed, V) in enumerate(valid_seeds):
        for pid in V:
            need_map.setdefault(pid, []).append(idx)

    # Prepare output files
    # One jsonl per subgraph containing full records
    writers = []
    for i, (seed, V) in enumerate(valid_seeds):
        fp = (out_dir / f"subgraph_{i:03d}.jsonl").open("w", encoding="utf-8")
        writers.append(fp)

    # Optional: also save a small meta file per subgraph (seed + size),方便你查
    meta = []
    for i, (seed, V) in enumerate(valid_seeds):
        meta.append({"file": f"subgraph_{i:03d}.jsonl", "seed": seed, "num_nodes": len(V)})
    (out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] PASS 2/2: scan again and dump full records into each subgraph jsonl...")

    # PASS 2: stream file again, write matching records
    seen = 0
    written_counts = [0] * len(valid_seeds)
    # dedup: if the dataset has duplicate paper records with same id, avoid writing twice per subgraph
    written_ids_per_subgraph: List[Set[str]] = [set() for _ in range(len(valid_seeds))]

    t0 = time.time()
    for obj in iter_jsonl(in_path):
        seen += 1
        if not is_paper(obj):
            continue
        pid = obj.get("id")
        if pid not in need_map:
            continue

        line = json.dumps(obj, ensure_ascii=False)
        for sub_idx in need_map[pid]:
            if pid in written_ids_per_subgraph[sub_idx]:
                continue
            writers[sub_idx].write(line + "\n")
            written_ids_per_subgraph[sub_idx].add(pid)
            written_counts[sub_idx] += 1

        if seen % 500_000 == 0:
            elapsed = time.time() - t0
            sample = written_counts[:5]
            print(f"[PASS2] seen_objs={seen:,} time={elapsed:.1f}s sample_written(first5)={sample}")

    for fp in writers:
        fp.close()

    # Final check: some subgraphs might be missing nodes if dataset has holes (理论上你说不会)
    bad = 0
    for i, (seed, V) in enumerate(valid_seeds):
        if written_counts[i] != len(V):
            bad += 1
            print(f"[WARN] subgraph_{i:03d}: expected_nodes={len(V)} written={written_counts[i]} (missing {len(V)-written_counts[i]}) seed={seed}")
    if bad == 0:
        print("[DONE] All valid subgraphs exported with full records.")
    else:
        print(f"[DONE] Exported, but {bad} subgraphs have missing records (check WARN lines).")

    print(f"[DONE] out_dir={out_dir}")

if __name__ == "__main__":
    main()
