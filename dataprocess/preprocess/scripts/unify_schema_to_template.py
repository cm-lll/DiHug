from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any


TEMPLATE_SCHEMA: Dict[str, List[str]] = {
    "Author": ["HighImpact", "MidImpact", "LowImpact", "Other"],
    "Organization": ["University", "Company", "ResearchInstitute", "Other"],
    "Paper": ["Theory", "Application", "Survey", "System", "Experiment", "Other"],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_subtype_value(v) -> str:
    """Return a single subtype string (not list), normalized."""
    if v is None:
        return "Other"
    if isinstance(v, list):
        v = v[0] if len(v) > 0 else None
    if v is None:
        return "Other"
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return "Other"
    return s


def unify_one_subgraph(subdir: Path, template: Dict[str, List[str]], backup: bool) -> Dict[str, int]:
    """
    Returns stats:
      - changed_nodes
      - total_nodes
      - changed_schema (0/1)
    """
    stats = {"changed_nodes": 0, "total_nodes": 0, "changed_schema": 0}

    schema_path = subdir / "schema_by_type.json"
    typed_nodes_dir = subdir / "typed_nodes"

    if not typed_nodes_dir.exists():
        return stats

    # backup (optional)
    if backup:
        bdir = subdir / "_backup_before_unify_schema"
        bdir.mkdir(parents=True, exist_ok=True)
        if schema_path.exists():
            shutil.copy2(schema_path, bdir / "schema_by_type.json")
        for f in typed_nodes_dir.glob("*.jsonl"):
            shutil.copy2(f, bdir / f.name)

    # overwrite schema_by_type.json
    schema_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["changed_schema"] = 1

    # normalize typed_nodes/*.jsonl
    for f in typed_nodes_dir.glob("*.jsonl"):
        node_type = f.stem
        if node_type not in template:
            # unknown type: skip (or you can choose to set subtype="Other")
            continue

        allowed = set(template[node_type])
        rows = read_jsonl(f)
        out_rows = []
        for r in rows:
            stats["total_nodes"] += 1
            st = normalize_subtype_value(r.get("subtype"))
            if st not in allowed:
                st = "Other"
            # keep as list, consistent with your data format
            new_sub = [st]
            old_sub = r.get("subtype")
            # treat normalized comparison
            old_norm = normalize_subtype_value(old_sub)
            if old_norm != st:
                stats["changed_nodes"] += 1
            r["subtype"] = new_sub
            out_rows.append(r)

        write_jsonl(f, out_rows)

    return stats


def list_subgraph_dirs(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("subgraph_")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g. preprocess/output/ACM_subgraphs")
    ap.add_argument("--backup", action="store_true", help="Backup schema_by_type.json and typed_nodes/*.jsonl per subgraph")
    ap.add_argument("--dry_run", action="store_true", help="Only report what would change (no write)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    subdirs = list_subgraph_dirs(data_root)
    if not subdirs:
        raise RuntimeError(f"No subgraph dirs found under: {data_root}")

    total_changed = 0
    total_nodes = 0
    touched = 0

    for sd in subdirs:
        if args.dry_run:
            # dry_run: read + compute stats but do not write
            # We simulate by calling unify_one_subgraph on copies in memory:
            # For simplicity, we only report schema overwrite and subtype out-of-schema counts.
            typed_nodes_dir = sd / "typed_nodes"
            if not typed_nodes_dir.exists():
                continue
            would_change = 0
            would_total = 0
            for f in typed_nodes_dir.glob("*.jsonl"):
                node_type = f.stem
                if node_type not in TEMPLATE_SCHEMA:
                    continue
                allowed = set(TEMPLATE_SCHEMA[node_type])
                for r in read_jsonl(f):
                    would_total += 1
                    st = normalize_subtype_value(r.get("subtype"))
                    if st not in allowed:
                        would_change += 1
            print(f"[dry_run] {sd.name}: would_change_nodes={would_change} / total_nodes={would_total} (schema overwrite: yes)")
            total_changed += would_change
            total_nodes += would_total
            touched += 1
        else:
            st = unify_one_subgraph(sd, TEMPLATE_SCHEMA, backup=args.backup)
            print(f"[ok] {sd.name}: changed_nodes={st['changed_nodes']} / total_nodes={st['total_nodes']} (schema overwritten)")
            total_changed += st["changed_nodes"]
            total_nodes += st["total_nodes"]
            touched += 1

    print(f"\n[summary] subgraphs={touched} total_nodes={total_nodes} changed_nodes={total_changed}")
    if args.dry_run:
        print("[summary] dry_run only; no files were written.")


if __name__ == "__main__":
    main()
