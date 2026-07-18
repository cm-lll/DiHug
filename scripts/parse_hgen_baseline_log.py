#!/usr/bin/env python3
"""Parse one HGEN split-evaluation log into a machine-readable report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def last_float(text: str, pattern: str) -> float:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        raise ValueError(f"missing HGEN metric: {pattern}")
    return float(matches[-1].replace("−", "-"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")
    number = r"([-+−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    payload = {
        "format": "dihug-local-hgen-split-log-v1",
        "dataset": args.dataset,
        "source_log": str(args.log),
        "real": {
            "clustering": last_float(text, rf"Clustering coefficient ratio for real graph:\s*{number}"),
            "triangles": last_float(text, rf"Triangle count for real graph:\s*{number}"),
            "lcc": last_float(text, rf"LCC for real graph:\s*{number}"),
            "powerlaw": last_float(text, rf"Powerlaw Distribution Coeff\. for real Graph[^\d+−-]*{number}"),
            "assortativity": last_float(text, rf"Degree Assortativity\s+real graph:\s*{number}"),
        },
        "generated": {
            "clustering": last_float(text, rf"Clustering coefficient ratio for generated graph:\s*{number}"),
            "triangles": last_float(text, rf"Triangle count for generated graph:\s*{number}"),
            "lcc": last_float(text, rf"LCC for generated graph:\s*{number}"),
            "powerlaw": last_float(text, rf"Powerlaw Distribution Coeff\. for generated Graph[^\d+−-]*{number}"),
            "assortativity": last_float(text, rf"Degree Assortativity\s+generated graph:\s*{number}"),
        },
        "degree_mmd": last_float(
            text, rf"MMD distance for node degree distribution:\s*{number}"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
