import os,time,json,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/toy_gradient_alignment_20261111'
assert not OUT.exists(),'refuse overwrite'
start=time.time();peak=0;reason=None
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=subprocess.Popen([__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable),str(Path(__file__).with_name('run.py'))],cwd=ROOT,env=env,start_new_session=True)
while p.poll() is None:
    try:peak=max(peak,int(next(s.split()[1] for s in Path(f'/proc/{p.pid}/status').read_text().splitlines() if s.startswith('VmRSS:')))/1024)
    except (FileNotFoundError,StopIteration):pass
    if peak>2048:reason='memory limit'
    if time.time()-start>630:reason='wall limit'
    if reason:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
    time.sleep(.5)
if OUT.exists():(OUT/'supervisor.json').write_text(json.dumps(dict(seconds=time.time()-start,peak_RSS_MiB=peak,stop_reason=reason,exit_code=p.returncode,reaped=True,GPU_hours=0),indent=2))
