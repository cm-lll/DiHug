"""Independent wall/occupied-GPU bound. Only verified worker process groups."""
import os,time,json,signal
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_20261111'
w=json.loads((OUT/'window.json').read_text())
while True:
    r=json.loads((OUT/'registry.json').read_text())
    if r['done']:break
    used=r['preflight_GPU_seconds']+sum((a['end_epoch'] or time.time())-a['start_epoch'] for a in r['tasks'])
    if time.time()>w['science_stop_epoch'] or used>w['GPU_seconds_cap']-120:
        stopped=[]
        for a in r['tasks']:
            if a['end_epoch'] is not None:continue
            try:
                p=psutil.Process(a['pid']);cmd=p.cmdline()
                if not any('single_step_formal_20261111/worker.py' in s for s in cmd) or abs(p.create_time()-a['start_epoch'])>10:continue
                os.killpg(p.pid,signal.SIGTERM);stopped.append(p.pid)
            except psutil.NoSuchProcess:pass
        (OUT/'watchdog_stop.json').write_text(json.dumps(dict(epoch=time.time(),GPU_hours=used/3600,pids=stopped)))
        time.sleep(15)
        for pid in stopped:
            try:
                p=psutil.Process(pid)
                if any('single_step_formal_20261111/worker.py' in s for s in p.cmdline()):os.killpg(pid,signal.SIGKILL)
            except psutil.NoSuchProcess:pass
        break
    time.sleep(5)
