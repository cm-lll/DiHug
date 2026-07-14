#!/usr/bin/env python3
"""Summarize switch-space diffusion runs below one result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--real-triangles", type=float, default=655.0)
    parser.add_argument("--real-clustering", type=float, default=0.4689355333)
    args = parser.parse_args()

    header = (
        "run\tmode\ttop1\ttop5\tinverse_oracle\tmodel_oracle\t"
        "triangles\ttri_rel_err\tclustering\tclust_rel_err\tlcc\tfeature_mae"
    )
    print(header)
    for report_path in sorted(args.root.glob("**/sampling_report.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        final = payload["final_metrics"]
        train = payload.get("train_report", {})
        trace = payload.get("trace", [])
        feature_mae = (
            trace[-1].get("target_feature_mae", float("nan"))
            if trace
            else float("nan")
        )
        row = (
            str(report_path.parent.relative_to(args.root)),
            str(payload.get("sampling_mode", "model")),
            float(train.get("validation_top1", float("nan"))),
            float(train.get("validation_top5", float("nan"))),
            float(train.get("validation_oracle_top1", float("nan"))),
            float(payload.get("oracle_agreement", float("nan"))),
            float(final["triangles"]),
            abs(float(final["triangles"]) - args.real_triangles) / args.real_triangles,
            float(final["clustering"]),
            abs(float(final["clustering"]) - args.real_clustering) / args.real_clustering,
            float(final["lcc"]),
            float(feature_mae),
        )
        print(
            f"{row[0]}\t{row[1]}\t{row[2]:.4f}\t{row[3]:.4f}\t"
            f"{row[4]:.4f}\t{row[5]:.4f}\t{row[6]:.0f}\t{row[7]:.4f}\t"
            f"{row[8]:.6f}\t{row[9]:.4f}\t{row[10]:.0f}\t{row[11]:.6f}"
        )


if __name__ == "__main__":
    main()
