import os,sys,time,json,subprocess,signal
from pathlib import Path
H=Path(__file__).resolve().parent;O=Path(sys.argv[1]);p=json.loads((O/'protocol.json').read_text())
if (O/'lease.json').exists():raise SystemExit('duplicate launch refused')
snap=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
assert snap.split(',')[0]=='GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7' and int(snap.split(',')[1])<100
assert not subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip()
start=time.time();end=min(p['science_stop_epoch'],start+p['gpu_seconds_limit'])
if end-start<400:raise SystemExit('insufficient remaining time for frozen panel')
lease=dict(start_epoch=start,hard_stop_epoch=end,snapshot=snap)
def save():(O/'lease.json').write_text(json.dumps(lease,indent=2))
save();env=dict(os.environ,OBJECTIVE_DIAG_OUT=str(O),CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
try:
 with (O/'worker.log').open('w') as f:
  worker=subprocess.Popen([sys.executable,str(H/'run.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);lease['pid']=worker.pid;save()
  while worker.poll() is None and time.time()<end-15:time.sleep(1)
  if worker.poll() is None:
   os.killpg(worker.pid,signal.SIGTERM)
   try:worker.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
  lease['returncode']=worker.returncode
finally:lease.update(end_epoch=time.time(),gpu_hours=(time.time()-start)/3600);save()
print(json.dumps(lease),flush=True)
