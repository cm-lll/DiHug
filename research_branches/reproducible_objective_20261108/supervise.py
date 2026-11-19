"""Independent watchdog for this one new diagnostic lease only."""
import os, sys, time, json, subprocess, signal
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]).resolve()
protocol=json.loads((OUT/'protocol.json').read_text())
if (OUT/'lease.json').exists(): raise SystemExit('refuse duplicate launch')
start=time.time()
end=min(protocol['science_stop_epoch'],start+900-protocol.get('previous_gpu_seconds',0))
if end-start<600: raise SystemExit('insufficient remaining budget for full panel')
gpu='0'
snapshot=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip()
assert snapshot.split(',')[0]=='GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7'
assert int(snapshot.split(',')[1])<100, 'GPU occupied'
apps=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not apps, 'GPU compute process present'
lease=dict(start_epoch=start,hard_stop_epoch=end,physical_gpu=0,device_snapshot=snapshot,returncode=None)
def save(): (OUT/'lease.json').write_text(json.dumps(lease,indent=2))
save()
env=dict(os.environ,OBJECTIVE_DIAG_OUT=str(OUT),CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
try:
    with (OUT/'worker.log').open('w') as f:
        p=subprocess.Popen([sys.executable,str(HERE/'run.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        lease['pid']=p.pid;save()
        while p.poll() is None and time.time()<end-15: time.sleep(1)
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        lease['returncode']=p.returncode
finally:
    lease.update(end_epoch=time.time(),gpu_hours=(time.time()-start)/3600);save()
print(json.dumps(lease),flush=True)
