"""Independent owned-process watchdog; no other GPU process is signalled."""
import json,os,signal,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);p=json.loads((out/'protocol.json').read_text())
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
(out/'gpu_before.txt').write_text(rows)
for i in [0,1]:
    r=[x.strip() for x in rows.splitlines()[i].split(',')];assert int(r[0])==i and int(r[1])==0 and int(r[2])==0,'assigned GPU not idle'
procs=[]
for gpu,ds in [(0,'IMDB'),(1,'PubMed')]:
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=(out/(ds+'.log')).open('w');started=time.time()
    proc=subprocess.Popen([__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable),str(Path(__file__).with_name('worker.py')),str(out),ds],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    procs.append(dict(proc=proc,start=started,ds=ds,gpu=gpu,end=None,log=log))
(out/'launched.json').write_text(json.dumps([dict(dataset=r['ds'],gpu=r['gpu'],pid=r['proc'].pid,start_epoch=r['start']) for r in procs],indent=2))
while True:
    now=time.time()
    for r in procs:
        if r['end'] is None and r['proc'].poll() is not None:r['end']=now
    occupied=sum((r['end'] or now)-r['start'] for r in procs)
    (out/'watchdog.json').write_text(json.dumps(dict(at_epoch=now,occupied_seconds=occupied,processes=[dict(dataset=r['ds'],pid=r['proc'].pid,returncode=r['proc'].poll(),end_epoch=r['end']) for r in procs])))
    if all(r['end'] is not None for r in procs):break
    if now>=p['gpu_stop_epoch'] or occupied>=p['gpu_hours']*3600:
        for r in procs:
            if r['end'] is None:os.killpg(r['proc'].pid,signal.SIGTERM)
        time.sleep(5)
        for r in procs:
            if r['proc'].poll() is None:os.killpg(r['proc'].pid,signal.SIGKILL)
    time.sleep(2)
for r in procs:r['log'].close()
(out/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
print(json.dumps(dict(occupied_gpu_hours=occupied/3600,finished_epoch=time.time())))
