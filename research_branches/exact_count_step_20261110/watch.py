"""Own-process GPU lease watchdog; total wall deadline is never reset."""
import os,sys,json,time,subprocess,signal,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/('exact_count_step_20261110'+os.environ.get('EXACT_STEP_SUFFIX',''))
budget=json.loads((OUT/'budget.json').read_text());now=time.time()
remaining=budget['final_deadline_epoch']-budget['science_reserve_for_reporting_seconds']-now
if remaining<180:raise RuntimeError('insufficient remaining bounded window; do not launch')
query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpu=next(x for x in query.stdout.splitlines() if x.split(',')[0].strip()=='0')
v=[x.strip() for x in gpu.split(',')]
if float(v[3])!=0 or float(v[4])>0:raise RuntimeError('GPU0 no longer empty; no preemption or substitution')
start=time.time();end=min(start+budget['GPU_seconds_max']-budget.get('prior_occupied_seconds',0),budget['final_deadline_epoch']-budget['science_reserve_for_reporting_seconds'])
receipt=dict(hostname=os.uname().nodename,gpu_line=gpu,start_epoch=start,hard_stop_epoch=end,
    launch_local=datetime.datetime.fromtimestamp(start,datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    GPU_seconds_cap=budget['GPU_seconds_max'],final_deadline_epoch=budget['final_deadline_epoch'],pid=None,status='running')
(OUT/'supervisor.json').write_text(json.dumps(receipt,indent=2))
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=v[1]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
with (OUT/'worker.log').open('w') as log:
    p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['pid']=p.pid;(OUT/'supervisor.json').write_text(json.dumps(receipt,indent=2))
    while p.poll() is None:
        if time.time()>=end:
            receipt['status']='budget_stop';os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            break
        time.sleep(1)
    receipt.update(returncode=p.wait(),end_epoch=time.time(),occupied_seconds=time.time()-start,GPU_hours=(time.time()-start)/3600)
    if receipt['status']=='running':receipt['status']='complete' if p.returncode==0 else 'worker_failed'
post=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
receipt['release_query']=post.stdout;receipt['release_query_returncode']=post.returncode
(OUT/'supervisor.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
sys.exit(p.returncode)
