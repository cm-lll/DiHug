# preprocess/scripts/preprocess_all.py
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from glob import glob
from pathlib import Path

from preprocess.config.settings import settings
from preprocess.src.graph.extract_graph import extract_nodes_and_edges
from preprocess.src.utils.io import read_jsonl_in_chunks, write_jsonl
from preprocess.src.pipeline import Pipeline

# =============================
# Global config
# =============================
CHUNK_SIZE = settings.CHUNK_SIZE
NUM_WORKERS = settings.NUM_WORKERS
SAMPLE_SIZE = int(os.getenv("ROLE0_SAMPLE_SIZE", "20"))
ROLE1_BATCH_SIZE = int(os.getenv("ROLE1_BATCH_SIZE", "10"))

RESET_SPLIT = os.getenv("RESET_SPLIT", "0") == "1"
RESET_ROLE0 = os.getenv("RESET_ROLE0", "0") == "1"
RESET_ROLE1 = os.getenv("RESET_ROLE1", "0") == "1"

# =============================
# Globals for workers
# =============================
SCHEMA_MAP = {}
WORKER_ROLE1 = None


# =============================
# Utils
# =============================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_unlink(p: str):
    if os.path.exists(p):
        os.remove(p)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def sample_nodes_from_type_file(type_file: str, k: int):
    out = []
    for obj in iter_jsonl(type_file):
        out.append(obj)
        if len(out) >= k:
            break
    return out


def read_nodes_in_chunks(type_file: str, chunk_size: int):
    buf = []
    for obj in iter_jsonl(type_file):
        buf.append(obj)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


# =============================
# Stage S: Split (unchanged)
# =============================
def split_raw_to_type_files(raw_input: str, out_dir: str, chunk_size: int, reset: bool):
    raw_nodes_dir = os.path.join(out_dir, "raw_nodes")
    ensure_dir(raw_nodes_dir)

    paper_path = os.path.join(raw_nodes_dir, "Paper.jsonl")
    author_path = os.path.join(raw_nodes_dir, "Author.jsonl")
    org_path = os.path.join(raw_nodes_dir, "Organization.jsonl")
    edges_path = os.path.join(out_dir, "edges.jsonl")

    if reset:
        for p in [paper_path, author_path, org_path, edges_path]:
            safe_unlink(p)

    seen_paper = set()
    seen_org = set()
    author_index = {}

    if not reset and os.path.exists(paper_path):
        for x in iter_jsonl(paper_path):
            seen_paper.add(x["id"])

    if not reset and os.path.exists(org_path):
        for x in iter_jsonl(org_path):
            seen_org.add(x["id"])

    if not reset and os.path.exists(author_path):
        for a in iter_jsonl(author_path):
            author_index[a["id"]] = {
                "type": "Author",
                "id": a["id"],
                "name": a.get("name", ""),
                "org_name": a.get("org_name", ""),
                "paper_ids": set(a.get("paper_ids", [])),
                "citation_sum": int(a.get("citation_sum", 0)),
            }

    for chunk in read_jsonl_in_chunks(raw_input, chunk_size):
        papers, orgs, edges = [], [], []

        for rec in chunk:
            nodes, es = extract_nodes_and_edges(rec)

            for n in nodes:
                t, nid = n["type"], n["id"]

                if t == "Paper" and nid not in seen_paper:
                    seen_paper.add(nid)
                    papers.append(n)

                elif t == "Organization" and nid not in seen_org:
                    seen_org.add(nid)
                    orgs.append(n)

                elif t == "Author":
                    a = author_index.get(nid)
                    if a is None:
                        author_index[nid] = {
                            "type": "Author",
                            "id": nid,
                            "name": n.get("name", ""),
                            "org_name": n.get("org_name", ""),
                            "paper_ids": set(n.get("paper_ids", [])),
                            "citation_sum": int(n.get("citation_sum", 0)),
                        }
                    else:
                        a["paper_ids"].update(n.get("paper_ids", []))
                        a["citation_sum"] += int(n.get("citation_sum", 0))

            edges.extend(es)

        if papers:
            write_jsonl(paper_path, papers, append=True)
        if orgs:
            write_jsonl(org_path, orgs, append=True)
        if edges:
            write_jsonl(edges_path, edges, append=True)

    safe_unlink(author_path)
    author_dump = []
    for a in author_index.values():
        author_dump.append({
            **a,
            "paper_ids": sorted(a["paper_ids"])
        })

    author_dump.sort(key=lambda x: x["id"])
    write_jsonl(author_path, author_dump, append=True)

    print("[Split] Completed:", raw_nodes_dir)
    return raw_nodes_dir, edges_path


# =============================
# Stage 0: Role0 (SKIP if exists)
# =============================
def run_role0_schema(raw_nodes_dir: str, out_dir: str):
    schema_path = os.path.join(out_dir, "schema_by_type.json")

    if os.path.exists(schema_path) and not RESET_ROLE0:
        print("[Role0] schema_by_type.json exists → skip LLM")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("[Role0] Running LLM")
    pipeline = Pipeline(settings.API_KEY, use_role0=True)
    schema_by_type = {}

    for p in Path(raw_nodes_dir).glob("*.jsonl"):
        samples = sample_nodes_from_type_file(str(p), SAMPLE_SIZE)
        if not samples:
            schema_by_type[p.stem] = None
            continue

        schema = pipeline.role0.discover_schema(p.stem, samples)
        if isinstance(schema, dict) and schema.get("should_classify"):
            schema_by_type[p.stem] = schema["schema"]
        else:
            schema_by_type[p.stem] = None

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_by_type, f, ensure_ascii=False, indent=2)

    return schema_by_type


# =============================
# Stage 1: Role1 (SKIP if exists)
# =============================
def init_worker_role1():
    global WORKER_ROLE1
    WORKER_ROLE1 = Pipeline(settings.API_KEY, use_role1=True).role1


def process_nodes_chunk_for_role1(args):
    nodes, node_type = args
    schema = SCHEMA_MAP.get(node_type)
    out = []

    for i in range(0, len(nodes), ROLE1_BATCH_SIZE):
        batch = nodes[i:i + ROLE1_BATCH_SIZE]
        labels_list = WORKER_ROLE1.extract_batch(batch, schema)

        for n, labels in zip(batch, labels_list):
            n2 = dict(n)
            n2["subtype"] = labels
            out.append(n2)

    return out


def run_role1_by_type(raw_nodes_dir: str, out_dir: str, schema_by_type: dict):
    global SCHEMA_MAP
    SCHEMA_MAP = schema_by_type

    typed_dir = os.path.join(out_dir, "typed_nodes")
    ensure_dir(typed_dir)

    for p in Path(raw_nodes_dir).glob("*.jsonl"):
        node_type = p.stem
        out_file = os.path.join(typed_dir, f"{node_type}.jsonl")

        if os.path.exists(out_file) and not RESET_ROLE1:
            print(f"[Role1] {node_type} exists → skip LLM")
            continue

        print(f"[Role1] Running LLM for type={node_type}")
        safe_unlink(out_file)

        tasks = ((chunk, node_type) for chunk in read_nodes_in_chunks(str(p), CHUNK_SIZE))

        with Pool(NUM_WORKERS, initializer=init_worker_role1) as pool:
            for labeled in pool.imap(process_nodes_chunk_for_role1, tasks):
                write_jsonl(out_file, labeled, append=True)


# =============================
# Main
# =============================
if __name__ == "__main__":

    RAW_INPUT_DIR = os.getenv("RAW_INPUT_DIR", "")
    RAW_INPUT = os.getenv("RAW_INPUT", settings.RAW_INPUT)
    OUT_DIR = os.getenv("OUT_DIR", "preprocess/output")
    LIMIT = int(os.getenv("LIMIT", "0"))
    PATTERN = os.getenv("PATTERN", "subgraph_*.jsonl")

    ensure_dir(OUT_DIR)

    if RAW_INPUT_DIR:
        files = sorted(glob(os.path.join(RAW_INPUT_DIR, PATTERN)))
        if LIMIT > 0:
            files = files[:LIMIT]

        for f in files:
            base = Path(f).stem
            out_dir = os.path.join(OUT_DIR, base)
            ensure_dir(out_dir)

            raw_nodes_dir, _ = split_raw_to_type_files(
                raw_input=f,
                out_dir=out_dir,
                chunk_size=CHUNK_SIZE,
                reset=RESET_SPLIT
            )

            schema_by_type = run_role0_schema(raw_nodes_dir, out_dir)
            run_role1_by_type(raw_nodes_dir, out_dir, schema_by_type)

        print("\n========== ALL DATASETS COMPLETED ==========")
        raise SystemExit(0)

    # single file mode
    raw_nodes_dir, _ = split_raw_to_type_files(
        RAW_INPUT, OUT_DIR, CHUNK_SIZE, RESET_SPLIT
    )
    schema_by_type = run_role0_schema(raw_nodes_dir, OUT_DIR)
    run_role1_by_type(raw_nodes_dir, OUT_DIR, schema_by_type)

    print("\n========== PIPELINE COMPLETED ==========")
