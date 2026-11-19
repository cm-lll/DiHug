"""Independent single-worker wall/RSS watchdog; no GPU query or allocation."""
import os,sys,time,json,subprocess,signal
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[2];out=root/'results/fixed_count_diagnostic_20261110'
start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=root/'results/fixed_count_diagnostic_20261110_worker.log'
with log.open('x') as f:
    p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    peak=0;reason=None
    while p.poll() is None:
        try:rss=sum(x.memory_info().rss for x in [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True) if x.is_running())
        except psutil.NoSuchProcess:rss=0
        peak=max(peak,rss)
        if time.time()-start>600 or rss>2*1024**3:
            reason='wall' if time.time()-start>600 else 'RSS';os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            break
        time.sleep(1)
    p.wait()
out.mkdir(exist_ok=True)
receipt=dict(start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,returncode=p.returncode,stop_reason=reason,peak_RSS_MiB=peak/2**20,GPU_hours=0,worker_pid=p.pid,worker_exited=True)
(out/'supervisor.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
