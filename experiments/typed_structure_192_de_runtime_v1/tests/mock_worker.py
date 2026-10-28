"""CPU process fixture, NOT a scientific worker and no activation authority."""
import argparse
import json
import os
import pathlib
import time

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--launch-fd',type=int,required=True);p.add_argument('--behavior',required=True);a,unused=p.parse_known_args()
out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
ticket=json.loads(os.read(a.launch_fd,16384));os.close(a.launch_fd)
(out/'fixture_ticket.json').write_text(json.dumps(ticket))
if a.behavior=='sleep':time.sleep(10)
rc=9 if a.behavior=='fail' else 0
(out/'summary.json').write_text(json.dumps(dict(status='complete' if rc==0 else 'failed_or_incomplete',completed_updates=1000 if rc==0 else 0,completed_chains=2 if rc==0 else 0,error=None if rc==0 else 'synthetic_worker_failure',fixture_only=True,actual_optimizer_steps=0,actual_generated_graphs=0)))
raise SystemExit(rc)
