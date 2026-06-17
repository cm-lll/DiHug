#!/usr/bin/env python3
import argparse
import ast
import csv
import os
import re
from typing import Dict, List, Optional


METRIC_KEYS = [
    "test/EdgeTypesTV",
    "test/Graph/Clustering_real",
    "test/Graph/Clustering_gen",
    "test/Graph/Triangles_real",
    "test/Graph/Triangles_gen",
    "test/Graph/LCCSize_real",
    "test/Graph/LCCSize_gen",
    "test/Graph/EdgeOverlapRate",
    "test/Graph/PowerLawAlpha_real",
    "test/Graph/PowerLawAlpha_gen",
    "test/Graph/DegreeAssortativity_real",
    "test/Graph/DegreeAssortativity_gen",
    "test/Graph/DegreeMMD",
]


def extract_overall_dict(text: str) -> Optional[Dict]:
    marker = "For overall "
    idx = text.rfind(marker)
    if idx == -1:
        return None
    tail = text[idx:]
    # First Python dict literal after "For overall ...".
    m = re.search(r"\{.*\}", tail, flags=re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except Exception:
        return None


def normalize_value(v):
    # Keep "(mean, std)" tuples as-is; scalar stays scalar.
    return v


def summarize_logs(log_paths: List[str]) -> List[Dict]:
    rows = []
    for path in log_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        d = extract_overall_dict(text)
        row = {"log_file": os.path.basename(path), "status": "ok" if d else "missing_overall"}
        if d:
            for k in METRIC_KEYS:
                row[k] = normalize_value(d.get(k))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize test metrics from training logs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Log file paths (supports shell expansion from caller).",
    )
    parser.add_argument(
        "--out_csv",
        default="output/test_metrics_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = summarize_logs(args.logs)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    fieldnames = ["log_file", "status"] + METRIC_KEYS
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote summary: {args.out_csv}")
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Parsed logs: {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
