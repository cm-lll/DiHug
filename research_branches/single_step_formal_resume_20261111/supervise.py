"""One frozen queue, occupied-device accounting, CPU evaluation after release."""
import os,sys,time,json,signal,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_resume_20261111';HERE=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text())
def save(n,x):
    p=OUT/n;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def kill(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
    start=time.time();p=read('protocol.json');window=read('window.json');pre=read('preflight_receipt.json');assert pre['passed'];pre['GPU_seconds']+=window['prior_GPU_seconds'];tasks=p['tasks'];cal={}
    for idx in (0,1,4,5):
        t=tasks[idx];v=read('preflight_'+t['id']+'/gates.json');assert v['passed'];warm=v['updates'][-1]['seconds'];sample=v['sampling']['seconds']/5
        # Warm cost plus conservative concurrency margin; setup/save separately.
        cal[t['parent']['dataset'],t['B']]=dict(update_seconds=warm,reverse_step_seconds=sample,estimated_seconds=1.4*(500*warm+4000*sample+180))
    estimates=[cal[t['parent']['dataset'],t['B']]['estimated_seconds'] for t in tasks];estimated_gpu=pre['GPU_seconds']+sum(estimates);need=max(estimates)
    fits=(time.time()+need+120<window['science_stop_epoch'] and estimated_gpu<window['GPU_seconds_cap'])
    save('budget_calibration.json',dict(prior_GPU_seconds=window['prior_GPU_seconds'],new_preflight_GPU_seconds=pre['GPU_seconds']-window['prior_GPU_seconds'],occupied_GPU_seconds_before_formal=pre['GPU_seconds'],per_task=[dict(task=t['id'],**cal[t['parent']['dataset'],t['B']]) for t in tasks],predicted_GPU_hours=estimated_gpu/3600,predicted_parallel_minutes=need/60,remaining_science_minutes=(window['science_stop_epoch']-time.time())/60,fits=fits,factor=1.4,chain_time_extrapolated=True,not_guaranteed='five steps do not bound late-chain density, concurrency or CPU evaluation costs'))
    if not fits:save('launch_decision.json',dict(started=False,reason='full panel does not fit measured budget'));return
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader'],text=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);dev={int(x.split(',')[0]):[y.strip() for y in x.split(',')] for x in smi.splitlines()}
    assert all(i in dev and '2080 Ti' in dev[i][2] and int(dev[i][3].split()[0])<100 and dev[i][1] not in apps for i in range(8)),'eight GPUs not idle; no launch'
    files=list(HERE.glob('*.py'));save('execution_bindings.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files});records=[];live={};cpu={};waiting=[];watch=None
    def publish():save('registry.json',dict(supervisor_pid=os.getpid(),preflight_GPU_seconds=pre['GPU_seconds'],tasks=records,done=False))
    try:
        publish()
        with (OUT/'watchdog.log').open('x') as log:watch=subprocess.Popen([sys.executable,str(HERE/'watchdog.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for idx,t in enumerate(tasks):
            tick=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(t['gpu']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=(OUT/f'formal_{idx}.log').open('x');proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'formal',str(idx)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();live[idx]=proc;records.append(dict(index=idx,task=t['id'],gpu=t['gpu'],pid=proc.pid,start_epoch=tick,end_epoch=None));publish()
        save('launch_decision.json',dict(started=True,start_epoch=start,formal_start_epoch=records[0]['start_epoch'],tasks=records,science_stop_epoch=window['science_stop_epoch'],delivery_stop_epoch=window['delivery_stop_epoch'],GPU_hours_cap=16,prior_GPU_hours=window['prior_GPU_seconds']/3600,resumed_from=2500))
        while live or cpu or waiting:
            for idx,proc in list(live.items()):
                if proc.poll() is not None:
                    records[idx].update(end_epoch=time.time(),returncode=proc.returncode);del live[idx];waiting.append(idx);publish()
                    if proc.returncode:save(f'failure_notice_{idx}.json',dict(task=tasks[idx]['id'],returncode=proc.returncode,epoch=time.time(),receipt=str(OUT/tasks[idx]['id']/'receipt.json')))
            used=pre['GPU_seconds']+sum((r['end_epoch'] or time.time())-r['start_epoch'] for r in records)
            if live and (time.time()>window['science_stop_epoch'] or used>=window['GPU_seconds_cap']-120 or (OUT/'watchdog_stop.json').exists()):
                for idx,proc in list(live.items()):kill(proc);records[idx].update(end_epoch=time.time(),returncode=proc.returncode);waiting.append(idx)
                live.clear();publish()
            for idx,proc in list(cpu.items()):
                if proc.poll() is not None:records[idx]['evaluation_returncode']=proc.returncode;del cpu[idx];publish()
            while waiting and len(cpu)<4 and time.time()<window['delivery_stop_epoch']-90:
                idx=waiting.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';log=(OUT/f'evaluate_{idx}.log').open('x');cpu[idx]=subprocess.Popen([sys.executable,str(HERE/'evaluate.py'),str(idx)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
            if time.time()>window['delivery_stop_epoch']-60:
                for proc in cpu.values():kill(proc)
                cpu.clear();waiting.clear();break
            save('supervisor_progress.json',dict(epoch=time.time(),GPU_hours=used/3600,active_GPUs=list(live),CPU_evaluating=list(cpu),tasks=records))
            time.sleep(5)
    except BaseException as ex:save('supervisor_failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
    finally:
        for idx,proc in live.items():kill(proc);records[idx].update(end_epoch=time.time(),returncode=proc.returncode)
        for proc in cpu.values():kill(proc)
        save('registry.json',dict(supervisor_pid=os.getpid(),preflight_GPU_seconds=pre['GPU_seconds'],tasks=records,done=True))
        if watch is not None:
            try:watch.wait(timeout=15)
            except subprocess.TimeoutExpired:kill(watch)
        used=pre['GPU_seconds']+sum((r['end_epoch'] or time.time())-r['start_epoch'] for r in records);save('supervisor_receipt.json',dict(start_epoch=start,end_epoch=time.time(),GPU_hours=used/3600,tasks=records,all_ticket_GPU_processes_exited=True,after_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)))
        with (OUT/'finalize.log').open('w') as log:result=subprocess.run([sys.executable,str(HERE/'finalize.py'),'--out',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=45)
        save('finalization_receipt.json',dict(returncode=result.returncode,completion_exists=(OUT/'completion.json').exists(),epoch=time.time()))
if __name__=='__main__':main()
