"""Independent budget guard; kill only this ticket's verified process groups."""
import json,os,signal,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real192_update_audit_20261111'
w=json.loads((OUT/'window.json').read_text())
while True:
    r=json.loads((OUT/'registry.json').read_text())
    if r['done']:break
    used=sum((a['end_epoch'] or time.time())-a['start_epoch'] for a in r['tasks'])
    if time.time()>=w['science_stop_epoch'] or used>=w['GPU_seconds_cap']-60:
        killed=[]
        for a in r['tasks']:
            if a['end_epoch'] is not None:continue
            try:
                p=psutil.Process(a['pid'])
                if not any('real192_update_audit_20261111/worker.py' in s for s in p.cmdline()) or abs(p.create_time()-a['start_epoch'])>10:continue
                os.killpg(p.pid,signal.SIGTERM);killed.append(p.pid)
            except psutil.NoSuchProcess:pass
        (OUT/'watchdog_stop.json').write_text(json.dumps(dict(epoch=time.time(),GPU_hours=used/3600,pids=killed)))
        time.sleep(10)
        for pid in killed:
            try:
                p=psutil.Process(pid)
                if any('real192_update_audit_20261111/worker.py' in s for s in p.cmdline()):os.killpg(pid,signal.SIGKILL)
            except psutil.NoSuchProcess:pass
        break
    time.sleep(2)
