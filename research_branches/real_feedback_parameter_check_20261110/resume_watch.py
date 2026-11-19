"""Same scientific panel, one model/process; completed first model not rerun."""
import os,sys,time,subprocess,signal,json,shutil
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_feedback_parameter_check_20261110';OLD=OUT.with_name(OUT.name+'_attempt0_partial');prior=json.loads((OLD/'supervisor.json').read_text());origin=prior['start_epoch'];used=prior['elapsed_seconds'];records=[];conditions=json.loads((OUT/'conditions.json').read_text());reason=None
for seed in (2026110412,2026110421,2026110422):
    start=time.time()
    if start-origin>840 or used>630:reason='insufficient remaining fixed budget';break
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True);line=next(x for x in smi.splitlines() if x.startswith('0,'));a=[s.strip() for s in line.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if int(a[2].split()[0])>=100 or int(a[3].split()[0])>=5 or a[1] in apps:reason='GPU no longer idle';break
    task=OUT/f'model_{seed}';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',DIHUG_MODEL_SEED=str(seed),DIHUG_DIAG_OUT=str(task),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    alloc=time.time()
    with (OUT/f'model_{seed}.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);peak=0
        while p.poll() is None:
            try:rss=sum(x.memory_info().rss for x in [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True));peak=max(peak,rss)
            except psutil.NoSuchProcess:rss=0
            if time.time()-origin>840 or used+time.time()-alloc>718 or rss>4*1024**3:
                reason='wall/GPU time' if time.time()-origin>840 or used+time.time()-alloc>718 else 'RSS';os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(.5)
        p.wait()
    elapsed=time.time()-alloc;used+=elapsed;records.append(dict(seed=seed,pid=p.pid,seconds=elapsed,returncode=p.returncode,peak_RSS_MiB=peak/2**20))
    if (task/'conditions.json').exists():
        conditions.extend(json.loads((task/'conditions.json').read_text()));(OUT/'conditions.json').write_text(json.dumps(conditions,indent=2))
        for f in task.glob('*_grams.npz'):shutil.copy2(f,OUT/f.name)
    (OUT/'resume_progress.json').write_text(json.dumps(dict(tasks=records,cumulative_GPU_seconds=used,conditions=len(conditions)),indent=2))
    if p.returncode!=0:reason=reason or 'worker failure';break
post=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
r=dict(status='completed' if len(conditions)==12 else 'partial',original_start_epoch=origin,end_epoch=time.time(),wall_seconds=time.time()-origin,GPU_hours_conservative=used/3600,GPU_seconds=used,completed_conditions=len(conditions),tasks=records,prior_attempt=prior,stop_reason=reason,after_apps=post,all_workers_exited=True)
(OUT/'supervisor.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
