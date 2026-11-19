import os,sys,json,time,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_20261111';start=time.time();rows=[];reason=None
for idx in (0,1,4,5):
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True);line=next(a for a in smi.splitlines() if a.startswith('0,'));parts=[x.strip() for x in line.split(',')];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if int(parts[2].split()[0])>100 or parts[1] in apps:reason='GPU0 no longer idle';break
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');tick=time.time()
    with (OUT/f'preflight_{idx}.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),'preflight',str(idx)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    rows.append(dict(task_index=idx,pid=p.pid,returncode=p.returncode,allocation_epoch=tick,release_epoch=time.time(),occupied_seconds=time.time()-tick))
    if p.returncode:reason='preflight failure';break
receipt=dict(passed=len(rows)==4 and reason is None,tasks=rows,GPU_seconds=sum(x['occupied_seconds'] for x in rows),start_epoch=start,end_epoch=time.time(),stop_reason=reason,all_workers_exited=True)
(OUT/'preflight_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
