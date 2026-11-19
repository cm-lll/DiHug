import os,sys,time,json,subprocess,signal
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'results/direct_structure_cpu_20261108';p=json.loads((O/'protocol.json').read_text())
if (O/'execution.json').exists():raise SystemExit('duplicate launch refused')
start=time.time();deadline=min(p['science_stop_epoch'],start+540)
if deadline-start<180:raise SystemExit('insufficient panel time')
receipt=dict(start_epoch=start,hard_stop_epoch=deadline,CPU_threads=1,GPU=0)
def save():(O/'execution.json').write_text(json.dumps(receipt,indent=2))
save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (O/'worker.log').open('w') as f:
    worker=subprocess.Popen([sys.executable,str(H/'run.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);receipt['pid']=worker.pid;save()
    while worker.poll() is None:
        if time.time()>=deadline-10:receipt['stop_reason']='wall cutoff';break
        try:
            txt=Path(f'/proc/{worker.pid}/status').read_text();rss=int(next(x for x in txt.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
            if rss>p['max_RSS_bytes']:receipt['stop_reason']='RSS cutoff';break
        except (FileNotFoundError,StopIteration):pass
        time.sleep(1)
    if worker.poll() is None:
        os.killpg(worker.pid,signal.SIGTERM)
        try:worker.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
    receipt.update(returncode=worker.returncode,end_epoch=time.time(),seconds=time.time()-start);save()
print(json.dumps(receipt),flush=True)
