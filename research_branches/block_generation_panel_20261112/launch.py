"""Six-card ready queue; hard wall and allocated-card-hour enforcement."""
import os,sys,json,time,signal,subprocess,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def busy():
    q=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    return {r.split(',')[0].strip():float(r.split(',')[1]) for r in q.splitlines()}
def main():
    out=Path(sys.argv[1]).resolve();p=read(out/'protocol.json');queue=list(p['new_jobs']);active={};finished=[];started=[];halt=None
    def hours():return (sum(x['seconds'] for x in finished)+sum(time.time()-x['start_epoch'] for x in active.values()))/3600
    def terminate():
        for x in active.values():
            if x['process'].poll() is None:os.killpg(x['process'].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('supervisor terminated')))
    try:
        while queue or active:
            now=time.time()
            for uuid,x in list(active.items()):
                rc=x['process'].poll()
                if rc is not None:
                    x['log'].close();finished.append(dict(job=x['job']['id'],gpu_uuid=uuid,pid=x['process'].pid,exit_code=rc,
                        start_epoch=x['start_epoch'],end_epoch=now,seconds=now-x['start_epoch']))
                    del active[uuid]
                    if rc!=0:halt='worker failure; preserve evidence and stop new queue admissions'
            if now>=p['science_stop_epoch']-15:halt='science deadline'
            if hours()>=p['max_GPU_hours']-0.04:halt='allocated GPU-hour budget reserve'
            if halt:
                terminate()
                # Failed jobs are not silently retried. Preserve all pending jobs.
                for uuid,x in list(active.items()):
                    try:rc=x['process'].wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(x['process'].pid,signal.SIGKILL);rc=x['process'].wait()
                    x['log'].close();end=time.time();finished.append(dict(job=x['job']['id'],gpu_uuid=uuid,pid=x['process'].pid,exit_code=rc,start_epoch=x['start_epoch'],end_epoch=end,seconds=end-x['start_epoch']))
                    del active[uuid]
                break
            if queue:
                mem=busy()
                for dev in p['devices']:
                    uuid=dev['uuid']
                    if not queue:break
                    if uuid in active or mem.get(uuid,99999)>16:continue
                    job=queue.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
                    log=(out/'logs'/(job['id']+'.log')).open('w');ts=time.time()
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),str(out),job['id']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[uuid]=dict(process=proc,log=log,start_epoch=ts,job=job)
                    started.append(dict(job=job['id'],gpu_index=dev['index'],gpu_uuid=uuid,pid=proc.pid,start_epoch=ts))
                    save(out/'launches.json',started)
            save(out/'queue_status.json',dict(at_epoch=time.time(),allocated_GPU_hours=hours(),running=[dict(job=x['job']['id'],gpu_uuid=u,pid=x['process'].pid) for u,x in active.items()],pending=[j['id'] for j in queue],finished=finished,stop_reason=halt))
            if queue or active:time.sleep(2)
    except BaseException:
        halt=traceback.format_exc();terminate()
        for uuid,x in active.items():
            try:rc=x['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(x['process'].pid,signal.SIGKILL);rc=x['process'].wait()
            end=time.time();x['log'].close();finished.append(dict(job=x['job']['id'],gpu_uuid=uuid,pid=x['process'].pid,exit_code=rc,start_epoch=x['start_epoch'],end_epoch=end,seconds=end-x['start_epoch']))
        active.clear()
    allocation=sum(x['seconds'] for x in finished)/3600
    save(out/'GPU_release.json',dict(at_epoch=time.time(),allocated_GPU_hours=allocation,finished=finished,pending=[x['id'] for x in queue],stop_reason=halt,all_own_children_exited=True))
    # CPU evaluation starts only after all ticket GPU processes have exited.
    remain=p['delivery_epoch']-time.time()-20
    if remain>0:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (out/'evaluation.log').open('w') as log:
            evaluator=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate.py')),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:erc=evaluator.wait(timeout=remain)
            except subprocess.TimeoutExpired:os.killpg(evaluator.pid,signal.SIGTERM);erc=evaluator.wait(timeout=10)
    else:erc=None
    save(out/'completion.json',dict(at_epoch=time.time(),GPU_hours=allocation,evaluation_exit=erc,
        completed_jobs=sum(x['exit_code']==0 for x in finished),expected_new_jobs=len(p['new_jobs']),stop_reason=halt,
        pending=[x['id'] for x in queue],status='complete' if erc==0 and len(finished)==len(p['new_jobs']) and all(x['exit_code']==0 for x in finished) else 'partial',no_followup_started=True))
if __name__=='__main__':main()
