"""Reserve concurrent shutdown time before the absolute allocation cap.

Added as a separate CPU guard; no running worker/source is hot-replaced.
"""
import os,sys,time,json,signal,hashlib
from pathlib import Path
out=Path(sys.argv[1]).resolve()
p=json.loads((out/'protocol.json').read_text())
def write(name,x):
    q=out/name;t=q.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(q)
write('exit_guard_started.json',dict(pid=os.getpid(),epoch=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rule='Reserve 8 seconds per currently live allocation plus 2 GPU-seconds; scientific stop also has 8-second shutdown reserve.',GPU_hours=0))
while not (out/'gpu_finished.json').exists():
    rows=json.loads((out/'allocations.json').read_text());now=time.time()
    active=[r for r in rows if not r.get('end_epoch')]
    used=sum((r.get('end_epoch') or now)-r['start_epoch'] for r in rows)
    if active and (used+8*len(active)+2>=p['GPU_seconds_limit'] or now+8>=p['science_stop_epoch']):
        (out/'STOP').write_text('allocation shutdown reserve reached\n')
        write('exit_guard_stop.json',dict(epoch=now,GPU_seconds=used,active=len(active),shutdown_reserve_seconds=8))
        owned=[]
        for r in active:
            try:
                cmd=Path(f'/proc/{r["pid"]}/cmdline').read_bytes()
                if b'reliability_worker.py' in cmd and str(out).encode() in cmd and os.getpgid(r['pid'])==r['pid']:
                    owned.append(r['pid']);os.killpg(r['pid'],signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(3)
        for pid in owned:
            try:os.killpg(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        break
    time.sleep(1)
write('exit_guard_finished.json',dict(epoch=time.time(),GPU_hours=0))
