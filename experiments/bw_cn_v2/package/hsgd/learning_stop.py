"""Safe stop request for ONE named run. No PID killing or other-task signals."""
import argparse
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path);a=p.parse_args()
    out=a.directory.resolve()
    c=json.loads((out/"run_control.json").read_text())
    if c["version"]!="bw_cn_v2":raise RuntimeError("wrong run version")
    path=out/"STOP_REQUESTED"
    try:
        with path.open("x") as f:json.dump(dict(requested_epoch=time.time(),action="finish current safe boundary, final validation and atomic save"),f)
    except FileExistsError:pass
    print("Stop requested:",out,". Wait for stopped_manual and a valid final checkpoint; do not kill -9.")
    return 0


if __name__=="__main__":raise SystemExit(main())
