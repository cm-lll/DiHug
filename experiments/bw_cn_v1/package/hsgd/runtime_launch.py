"""stdlib-only outer cap including Torch imports, all children and verification.

Only manages the new process group created here. Does not touch any other job.
Use this entry on BW. Cleanup grace <=6s; cap <=570s, total <600s.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path,value):
    temp=path.with_name(path.name+".tmp");temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--wall-seconds",type=int,default=570)
    args,rest=p.parse_known_args()
    if not 30<=args.wall_seconds<=570:p.error("outer cap must be 30..570s, never extend automatically")
    out=args.output.resolve()
    if out.exists():raise RuntimeError("new independent output directory required")
    command=[sys.executable,"-m","hsgd.runtime_gate","--output",str(out),"--wall-seconds",str(args.wall_seconds-20),"--bounded-launch",*rest]
    print("Outer hard cap",args.wall_seconds,"s, cleanup <=6s; imports/startup/verification included.",flush=True)
    start=time.monotonic();child=subprocess.Popen(command,start_new_session=True)
    reason=None
    try:code=child.wait(timeout=args.wall_seconds)
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        reason=type(exc).__name__;os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
        code=75 if reason=="TimeoutExpired" else 130
    out.mkdir(parents=True,exist_ok=True)
    receipt=dict(child_exit_code=code,reason=reason,outer_wall_seconds=time.monotonic()-start,
        hard_cap_seconds=args.wall_seconds,cleanup_grace_seconds=6,managed_process_group=child.pid,
        automatic_restart=False,scope="complete controller including imports, workers and evidence validation")
    save(out/"outer_launcher.json",receipt)
    if reason:
        # Retain committed checkpoints. Do not leave a stale success claim on timeout.
        path=out/"summary.json"
        if path.exists():
            result=json.loads(path.read_text());result["statuses"]["next_stage_conditions"]="incomplete"
            result["outer_stop"]=receipt;save(path,result)
        print("Stopped own process group; committed artifacts retained; incomplete, no continuation.",flush=True)
    print(json.dumps(receipt,indent=2),flush=True)
    return code


if __name__=="__main__":sys.exit(main())
