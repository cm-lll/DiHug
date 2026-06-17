# scripts/pick_seeds_no_db.py
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Iterator, List, Set, Tuple

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
    # 只认顶层 paper：必须有 title/year/references(list)
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/pick_seeds_no_db.py <input_jsonl> <out_seeds_json>")
        sys.exit(1)

    in_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 你要 100 个子图，每个子图节点=1+indeg(seed)，所以 indeg ∈ [99, 499]
    INDEG_MIN, INDEG_MAX = 99, 499

    # 候选采样规模：越大越容易一次命中 100 个 seed
    # 50万通常够用（你文件 600万级），不够就加到 100万
    CANDIDATE_TARGET = 500_000

    # 为了可复现可固定随机种子
    # random.seed(1234)

    t0 = time.time()
    seen = 0
    candidates: List[str] = []
    candidate_set: Set[str] = set()
    indeg: Dict[str, int] = {}

    print(f"[INFO] Scanning once to sample ~{CANDIDATE_TARGET:,} candidate papers and count their indegree...")
    # 一遍扫描：前半段通过“概率采样”收集候选；全程对候选统计入度
    for obj in iter_jsonl(in_path):
        seen += 1
        if is_paper(obj):
            pid = obj["id"]

            # 采样候选：在早期快速装满，后期用概率维持规模（近似 reservoir）
            if len(candidates) < CANDIDATE_TARGET:
                candidates.append(pid)
                candidate_set.add(pid)
                indeg.setdefault(pid, 0)
            else:
                # 以小概率替换（让后面的 paper 也有机会进入候选）
                # p = CANDIDATE_TARGET / seen_papers 近似；这里用一个温和版本避免过慢
                if random.random() < 0.05:
                    # 替换一个随机候选
                    j = random.randrange(0, CANDIDATE_TARGET)
                    old = candidates[j]
                    candidates[j] = pid
                    candidate_set.discard(old)
                    candidate_set.add(pid)
                    indeg.pop(old, None)
                    indeg.setdefault(pid, 0)

            # 统计候选的入度：对该 paper 的 references 里如果命中候选，就给 ref +1
            refs = obj.get("references") or []
            for r in refs:
                if isinstance(r, str) and r in candidate_set:
                    indeg[r] = indeg.get(r, 0) + 1

        if seen % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"[PROG] seen_objs={seen:,} candidates={len(candidate_set):,} time={elapsed:.1f}s")

    # 筛 seed
    good = [(pid, d) for pid, d in indeg.items() if INDEG_MIN <= d <= INDEG_MAX]
    good.sort(key=lambda x: x[1])  # 按入度排序可选
    if len(good) < 100:
        print(f"[WARN] Only found {len(good)} seeds with indegree in [{INDEG_MIN},{INDEG_MAX}].")
        print("[WARN] Increase CANDIDATE_TARGET (e.g., 1_000_000) and rerun.")
    selected = good[:100]

    out = {
        "indeg_range": [INDEG_MIN, INDEG_MAX],
        "candidate_target": CANDIDATE_TARGET,
        "selected": [{"id": pid, "indeg": d} for pid, d in selected],
        "found": len(good),
        "seen_objects": seen
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.time() - t0
    print(f"[DONE] selected={len(selected)} found={len(good)} saved={out_path} time={elapsed:.1f}s")

if __name__ == "__main__":
    main()
