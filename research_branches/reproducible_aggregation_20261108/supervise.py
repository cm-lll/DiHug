"""Own-process wall/GPU lease watchdog; never touches unrelated processes."""
import os,sys,time,json,subprocess,signal
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=ROOT/'results/reproducible_aggregation_20261108T113027'
OUT=BASE/'attempt3';OUT.mkdir(exist_ok=True)
if (OUT/'lease.json').exists():raise SystemExit('refuse duplicate launch')
priors=[json.loads(p.read_text()) for p in [BASE/'lease.json',BASE/'attempt2/lease.json']]
assert all(p['returncode'] is not None for p in priors)
spent=sum(p['end_epoch']-p['start_epoch'] for p in priors)
begin=time.time();end=min(1794110127,begin+900-spent)
env=dict(os.environ,DIAG_OUTPUT=str(OUT),CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
lease=dict(start_epoch=begin,hard_stop_epoch=end,physical_gpu=0,uuid='GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7',scientific_stop_epoch=1794110127,final_deadline_epoch=1794110427)
(OUT/'lease.json').write_text(json.dumps(lease,indent=2))
with (OUT/'worker.log').open('w') as f:
    p=subprocess.Popen([sys.executable,str(HERE/'run_panel.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    lease['pid']=p.pid;(OUT/'lease.json').write_text(json.dumps(lease,indent=2))
    while p.poll() is None and time.time()<end-10:time.sleep(1)
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
lease.update(end_epoch=time.time(),returncode=p.returncode,gpu_hours=(time.time()-begin)/3600)
(OUT/'lease.json').write_text(json.dumps(lease,indent=2));print(json.dumps(lease),flush=True)
