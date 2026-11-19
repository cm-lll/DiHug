import os,time,json,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/toy_representability_20261110';OUT.mkdir(exist_ok=False)
start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');peak=0;why=None
(OUT/'budget.json').write_text(json.dumps(dict(start_epoch=start,science_seconds=540,watch_seconds=600,CPU_threads=1,RSS_MiB=2048,GPU_hours=0),indent=2))
with (OUT/'worker.log').open('w') as log:
    p=subprocess.Popen([__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable),str(Path(__file__).with_name('run.py'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while p.poll() is None:
        try:peak=max(peak,int(next(x.split()[1] for x in Path(f'/proc/{p.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))/1024)
        except (FileNotFoundError,StopIteration):pass
        if peak>2048:why='RSS_limit'
        if time.time()-start>600:why='wall_limit'
        if why:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
        time.sleep(.5)
    (OUT/'supervisor.json').write_text(json.dumps(dict(seconds=time.time()-start,end_epoch=time.time(),exit_code=p.returncode,stop_reason=why,peak_RSS_MiB=peak,GPU_hours=0,reaped=True),indent=2))
print((OUT/'supervisor.json').read_text())
