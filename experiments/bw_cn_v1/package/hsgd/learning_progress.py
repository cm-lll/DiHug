"""Read-only compact terminal progress/trend; no GPU/model calls."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);args=p.parse_args();out=args.directory
    b=json.loads((out/"run_control.json").read_text())
    print("created_utc:",b["created_utc"],"eight_hour_limit:",False,"hard_cutoff_utc:",None)
    print("stop_mode:",b["stop_mode"],"stop_after_updates:",b["stop_after_updates"])
    print("execution:","strict preflight; ordinary training AND validation")
    for name in ("launcher_progress.json","progress.json","latest.json"):
        path=out/name
        if path.exists():print(name,json.dumps(json.loads(path.read_text()),ensure_ascii=False))
    vals=list((out/"validation").glob("*/validation.json"))
    if vals:
        r=json.loads(max(vals,key=lambda p:p.stat().st_mtime).read_text())
        print("latest_validation:",json.dumps({k:r[k] for k in ("stage","completed_updates","pooled_ce","positive_ce","negative_ce","wall_seconds")},ensure_ascii=False))
        print("prior_CE:",r["baseline"]["pooled_ce"])
        print("expected_edges_by_t:",[(x["time"],x["expected_edges"]) for x in r["times"]])
    path=out/"summary.json"
    if path.exists():
        s=json.loads(path.read_text());print("terminal:",json.dumps({k:s.get(k) for k in ("status","reason","completed_updates","run_elapsed_seconds","final_checkpoint","stable_median_seconds","stable_p90_seconds","validation_trend","failure")},ensure_ascii=False))
    print("Same-reference denoising only. Stopping is not convergence/generation certification.")


if __name__=="__main__":main()
