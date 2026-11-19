import os,time,json,subprocess,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/('mean_anchored_toy_20261110'+('_attempt2' if len(sys.argv)>1 else ''));OUT.mkdir(exist_ok=False)
start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOY_OUT=str(OUT))
original_start=json.loads((ROOT/'results/mean_anchored_toy_20261110/budget.json').read_text())['start_epoch'] if len(sys.argv)>1 else start
(OUT/'budget.json').write_text(json.dumps(dict(start_epoch=start,worker_seconds=600,watch_seconds=660,delivery_seconds=900,CPU_threads=1,RSS_MiB=2048,GPU_hours=0),indent=2))
peak=0;reason=None
with (OUT/'worker.log').open('w') as log:
    p=subprocess.Popen([__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable),str(Path(__file__).with_name('run.py'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while p.poll() is None:
        try:peak=max(peak,int(next(x.split()[1] for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))/1024)
        except (FileNotFoundError,StopIteration):pass
        if peak>2048:reason='RSS_limit'
        if time.time()-original_start>660:reason='wall_limit'
        if reason:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
        time.sleep(.5)
result=dict(start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,worker_pid=p.pid,exit_code=p.returncode,stop_reason=reason,peak_RSS_MiB=peak,GPU_hours=0,reaped=True)
(OUT/'supervisor.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
