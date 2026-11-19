"""Dedicated one-GPU bounded supervisor. Never kills unrelated processes."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/same_input_update_check_20261110'
start=time.time();window=dict(start_epoch=start,science_stop_epoch=start+420,delivery_stop_epoch=start+600,GPU_stop_seconds=540)
(OUT/'window.json').write_text(json.dumps(window,indent=2));tasks=[];used=0.;reason=None
for i in range(4):
    if time.time()>window['science_stop_epoch'] or used>=420:reason='budget exhausted';break
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    row=next(x for x in smi.splitlines() if x.startswith('0,'));cols=[x.strip() for x in row.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if int(cols[2].split()[0])>=100 or int(cols[3].split()[0])>=5 or cols[1] in apps:reason='GPU0 not idle';break
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    alloc=time.time();peak=0
    with (OUT/f'worker_{i}.log').open('x') as f:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),str(i)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            try:peak=max(peak,sum(x.memory_info().rss for x in [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)))
            except psutil.NoSuchProcess:pass
            if time.time()>window['science_stop_epoch'] or used+time.time()-alloc>520:
                reason='time budget';os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(1)
        p.wait()
    elapsed=time.time()-alloc;used+=elapsed;tasks.append(dict(index=i,pid=p.pid,start_epoch=alloc,seconds=elapsed,returncode=p.returncode,peak_RSS_MiB=peak/2**20))
    (OUT/'supervisor_progress.json').write_text(json.dumps(dict(tasks=tasks,GPU_seconds=used),indent=2))
    if p.returncode:reason=reason or 'worker error';break
after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
receipt=dict(status='completed' if len(tasks)==4 and not reason else 'partial',start_epoch=start,end_epoch=time.time(),wall_seconds=time.time()-start,GPU_hours=used/3600,tasks=tasks,stop_reason=reason,after_apps=after,all_workers_exited=True)
(OUT/'supervisor.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)



