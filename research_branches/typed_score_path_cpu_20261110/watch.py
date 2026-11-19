"""Bounded CPU worker supervisor; only this child's PID may be terminated."""
import os,sys,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/typed_score_path_cpu_20261110'
start=time.monotonic();peak=0;reason=None
p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))])
while p.poll() is None:
    try:
        rss=next(int(x.split()[1])*1024 for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
        peak=max(peak,rss)
        if rss>2*1024**3:reason='resident memory limit'
    except (FileNotFoundError,ProcessLookupError,StopIteration):pass
    if time.monotonic()-start>240:reason='240s wall limit'
    if reason:
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        break
    time.sleep(.1)
out.mkdir(exist_ok=True)
(out/'watchdog.json').write_text(json.dumps(dict(returncode=p.wait(),reason=reason,wall_seconds=time.monotonic()-start,peak_sampled_RSS_MiB=peak/2**20,pid=p.pid),indent=2))
sys.exit(p.returncode)
