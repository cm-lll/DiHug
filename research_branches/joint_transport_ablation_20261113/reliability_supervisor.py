"""Bounded owned-process scheduler and independent allocation watchdog."""
import os,sys,time,json,signal,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/fullchain_reliability_run_20261113T133731'
START=1794548251
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
WORKER=Path(__file__).with_name('reliability_worker.py')
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def used(rows):return sum((r.get('end_epoch') or time.time())-r['start_epoch'] for r in rows)
def watchdog():
 while not (OUT/'gpu_finished.json').exists():
  try:rows=json.loads((OUT/'allocations.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
  if time.time()>=START+6000 or used(rows)>=9*3600-5:
   write(OUT/'watchdog_stop.json',dict(epoch=time.time(),GPU_seconds=used(rows),reason='science or cumulative GPU limit'))
   (OUT/'STOP').write_text('budget boundary\n')
   for r in rows:
    if r.get('end_epoch'):continue
    try:os.killpg(r['pid'],signal.SIGTERM)
    except ProcessLookupError:pass
   time.sleep(3)
   for r in rows:
    if r.get('end_epoch'):continue
    try:os.killpg(r['pid'],signal.SIGKILL)
    except ProcessLookupError:pass
   return
  time.sleep(1)
def main():
 OUT.mkdir(exist_ok=False)
 plan=ROOT/'results/fullchain_reliability_plan_20261113'
 bindings=json.loads((plan/'bindings.json').read_text())
 for p,h in bindings.items():assert sha(p)==h,p
 protocol=json.loads((plan/'protocol.json').read_text())
 protocol.update(status='authorized_execution',execution_authorized=True,no_actual_deadlines_bound=False,
  start_epoch=START,science_stop_epoch=START+6000,delivery_epoch=START+7200,GPU_seconds_limit=32400,
  original_plan_sha256=sha(plan/'protocol.json'),source_bindings={p:h for p,h in bindings.items() if p.endswith('.py')},
  execution_source_bindings={str(p):sha(p) for p in (WORKER,Path(__file__))})
 write(OUT/'protocol.json',protocol);write(OUT/'asset_recheck.json',dict(files=len(bindings),all_unchanged=True,epoch=time.time(),bindings=bindings))
 rows=[];write(OUT/'allocations.json',rows)
 wd=subprocess.Popen([PY,str(Path(__file__)),'watchdog'],start_new_session=True,stdout=(OUT/'watchdog.log').open('w'),stderr=subprocess.STDOUT)
 active={}
 def free(g):
  txt=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  if int(txt.strip())>100:raise RuntimeError(f'GPU{g} occupied: {txt}')
 def launch(mi,half,mode,g):
  free(g)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  tick=time.time();log=(OUT/f'worker_{mi}_{half}_{mode}.log').open('w')
  proc=subprocess.Popen([PY,str(WORKER),str(OUT),str(mi),str(half),mode],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  r=dict(model=mi,half=half,mode=mode,gpu=g,pid=proc.pid,start_epoch=tick,end_epoch=None)
  rows.append(r);active[proc.pid]=(proc,r,log);write(OUT/'allocations.json',rows)
 def drain():
  while active:
   for pid,(proc,r,log) in list(active.items()):
    rc=proc.poll()
    if rc is None:continue
    r.update(end_epoch=time.time(),returncode=rc);log.close();del active[pid];write(OUT/'allocations.json',rows)
   if active:time.sleep(1)
 try:
  for mi in range(4):launch(mi,0,'preflight',mi)
  drain()
  if any(r['returncode']!=0 for r in rows):raise RuntimeError('preflight failed; no new full collections')
  costs=[]
  for mi,row in enumerate(protocol['models']):
   d=OUT/row['id'];cal=json.loads((d/f'calibration_{mi}_0_preflight.json').read_text())
   grad=json.loads((d/'vector_00.json').read_text())
   costs.append(dict(model=row['id'],step=cal['seconds_per_step'],gradient_path=grad['seconds']))
  # Conservative bound includes initial-call warmup and 20% concurrent overhead.
  gpu_remaining=sum(9990*c['step']+15*c['gradient_path'] for c in costs)*1.2+8*30
  wall_remaining=max(5000*c['step']+8*c['gradient_path'] for c in costs)*1.2+120
  gate=used(rows)+gpu_remaining<32400 and time.time()+wall_remaining<START+6000
  write(OUT/'cost_gate.json',dict(costs=costs,GPU_hours_used=used(rows)/3600,predicted_remaining_GPU_hours=gpu_remaining/3600,
        predicted_remaining_science_minutes=wall_remaining/60,checked_epoch=time.time(),passed=gate,
        note='Ten real sampling steps and one actual eight-Xt gradient per parent; whole-chain and eight-card contention still extrapolated.'))
  if not gate:raise RuntimeError('actual throughput does not fit authorized complete panel')
  for mi in range(4):
   for half in range(2):launch(mi,half,'run',2*mi+half)
  drain()
  write(OUT/'gpu_finished.json',dict(epoch=time.time(),GPU_hours=used(rows)/3600,status='complete' if all(r['returncode']==0 for r in rows) else 'partial_failure'))
 except BaseException:
  write(OUT/'supervisor_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
  for proc,r,log in active.values():
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  drain();write(OUT/'gpu_finished.json',dict(epoch=time.time(),GPU_hours=used(rows)/3600,status='stopped'))
 finally:
  wd.wait(timeout=10)
  release=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  write(OUT/'release.json',dict(epoch=time.time(),all_owned_processes_exited=not active,GPU_hours=used(rows)/3600,host_compute_processes=release))
if __name__=='__main__':
 watchdog() if len(sys.argv)>1 and sys.argv[1]=='watchdog' else main()
