"""Independent wall/GPU lease bound; owns only its newly spawned worker."""
import os,sys,json,time,subprocess,signal,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_parameter_signal_20261109'
P=json.loads((OUT/'protocol.json').read_text());start=time.time()
if start>=P['science_stop_epoch']-60:raise SystemExit('Too late for bounded launch')
snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
(OUT/'resource_before.json').write_text(json.dumps(dict(gpus=snapshot.stdout,compute_apps=apps.stdout,time=start),indent=2))
line=next(s for s in snapshot.stdout.splitlines() if s.split(',')[0].strip()=='0');parts=[s.strip() for s in line.split(',')]
assert parts[1]==P['gpu_uuid'] and float(parts[3])==0 and P['gpu_uuid'] not in apps.stdout,'chosen card not free'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (OUT/'worker.log').open('x') as log:
 proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 lease=dict(start_epoch=start,pid=proc.pid,physical_gpu=0,gpu_uuid=P['gpu_uuid'],hard_stop=min(P['science_stop_epoch'],start+900),max_GPU_hours=.25)
 (OUT/'lease.json').write_text(json.dumps(lease,indent=2))
 timed_out=False
 try:rc=proc.wait(timeout=max(1,lease['hard_stop']-time.time()))
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
  try:rc=proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
 end=time.time();result=dict(exit_code=rc,timed_out=timed_out,lease_seconds=end-start,conservative_GPU_hours=(end-start)/3600,worker_exited=True)
 (OUT/'release.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
try:
 after=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 (OUT/'resource_after.json').write_text(json.dumps(dict(stdout=after.stdout,stderr=after.stderr,returncode=after.returncode),indent=2))
except subprocess.TimeoutExpired:(OUT/'resource_after.json').write_text(json.dumps(dict(error='post-release query timed out; worker exit confirmed')))
raise SystemExit(rc)
