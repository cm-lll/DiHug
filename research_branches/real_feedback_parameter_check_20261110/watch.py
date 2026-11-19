import os,sys,subprocess,time,signal,json
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_feedback_parameter_check_20261110';start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
# Only inspect devices; never stop another process. No allocation when resource is busy.
smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
line=next(x for x in smi.splitlines() if x.startswith('0,'));parts=[x.strip() for x in line.split(',')]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert int(parts[2].split()[0])<100 and int(parts[3].split()[0])<5 and parts[1] not in apps,'GPU0 not confirmed idle'
with (ROOT/'results/real_feedback_parameter_check_20261110_worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT);peak=0;reason=None
    while p.poll() is None:
        try:peak=max(peak,sum(x.memory_info().rss for x in [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)))
        except psutil.NoSuchProcess:pass
        if time.time()-start>720 or peak>4*1024**3:
            reason='time' if time.time()-start>720 else 'RSS';os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
            break
        time.sleep(.5)
    p.wait()
OUT.mkdir(exist_ok=True);elapsed=time.time()-start
post=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
r=dict(start_epoch=start,end_epoch=time.time(),elapsed_seconds=elapsed,GPU_hours_conservative=elapsed/3600,pid=p.pid,returncode=p.returncode,stop_reason=reason,peak_CPU_RSS_MiB=peak/2**20,allocation=line,before_apps=apps,after_apps=post,own_process_exited=True,preflight=smi)
(OUT/'supervisor.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
