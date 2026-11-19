"""Independent resident-memory/wall supervisor, never touches unrelated PIDs."""
import json,os,subprocess,sys,time
from pathlib import Path
start=time.monotonic(); peak=0; reason=None
p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('audit.py'))])
limit=float(os.environ.get('AUDIT_WALL_SECONDS','300'))
while p.poll() is None:
    try:
        fields=Path(f'/proc/{p.pid}/status').read_text().splitlines()
        rss=next(int(x.split()[1])*1024 for x in fields if x.startswith('VmRSS:'))
        peak=max(peak,rss)
        if rss>2*1024**3: reason='resident memory >2GiB'
    except (FileNotFoundError,StopIteration,ProcessLookupError): pass
    if time.monotonic()-start>limit: reason='wall limit'
    if reason:
        p.terminate()
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill();p.wait()
        break
    time.sleep(.1)
out=Path(__file__).resolve().parents[2]/'results'/('typed_score_cpu_20261110'+os.environ.get('AUDIT_RUN_SUFFIX',''))
out.mkdir(exist_ok=True)
(out/'watchdog.json').write_text(json.dumps(dict(pid=p.pid,returncode=p.wait(),reason=reason,wall_seconds=time.monotonic()-start,peak_sampled_RSS_MiB=peak/2**20,interval_seconds=.1),indent=2))
sys.exit(p.returncode)
