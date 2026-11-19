"""Finite queue supervisor. No adaptive experiments, retries, or quality selection."""
import os, sys, time, json, subprocess, signal
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)

def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)

def idle(gpus):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={int(x.split(',')[0]):[int(y.strip()) for y in x.split(',')[1:]] for x in text.strip().splitlines()}
    return all(rows[g][0]<=32 and rows[g][1]<=2 for g in gpus),rows

def main():
    out=Path(sys.argv[1]);p=json.loads((out/'protocol.json').read_text());budget=p['budgets']
    if (out/'window.json').exists():raise RuntimeError('window already exists; refuse duplicate launch')
    t=time.time();window=dict(start_epoch=t,science_stop_epoch=t+budget['science_minutes']*60,
                             delivery_stop_epoch=t+budget['wall_minutes']*60,GPU_seconds_limit=budget['GPU_hours']*3600,
                             supervisor_pid=os.getpid())
    save(out/'window.json',window);allocations=[];running={};watch=None
    def snapshot():save(out/'allocations.json',allocations)
    def launch(index,mode):
        row=p['tasks'][index];ok,inventory=idle([row['gpu']]);save(out/f'resource_{mode}_{index}.json',inventory)
        if not ok:raise RuntimeError('requested GPU no longer idle: '+str(row['gpu']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['gpu']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=open(out/f'{mode}_{index}.log','w');start=time.time()
        proc=subprocess.Popen([PY,str(HERE/'worker.py'),str(out),str(index),mode],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        a=dict(pid=proc.pid,start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21],gpu=row['gpu'],mode=mode,index=index,start_epoch=start,end_epoch=None)
        allocations.append(a);running[proc.pid]=(proc,a,log);snapshot()
    def wait_all():
        while running:
            spent=sum((a['end_epoch'] or time.time())-a['start_epoch'] for a in allocations)
            if time.time()>=window['science_stop_epoch']-30 or spent>=window['GPU_seconds_limit']-30 or (out/'STOP').exists():
                (out/'STOP').touch()
                for proc,a,log in running.values():
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            for pid,(proc,a,log) in list(running.items()):
                if proc.poll() is not None:
                    a.update(end_epoch=time.time(),returncode=proc.returncode);log.close();del running[pid];snapshot()
            time.sleep(2)
    try:
        ok,inventory=idle(range(8));save(out/'resource_start.json',inventory)
        if not ok:raise RuntimeError('eight requested devices not all idle; no task stealing')
        watch=subprocess.Popen([sys.executable,str(HERE/'watchdog.py'),str(out)],stdout=open(out/'watchdog.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
        # One matched CE/structural engineering pair per dataset; ephemeral states discarded.
        for i in (0,1,4,5):launch(i,'gate')
        wait_all()
        gates=[]
        for i in (0,1,4,5):
            f=out/('gate_'+p['tasks'][i]['task'])/'gate.json'
            gates.append(json.loads(f.read_text()) if f.exists() else dict(passed=False,index=i))
        save(out/'GPU_gate.json',dict(passed=all(x['passed'] for x in gates),gates=gates))
        if not all(x['passed'] for x in gates):
            save(out/'queue_result.json',dict(status='stopped_at_engineering_gate',formal_training=False,reason='gate failure; no panel launched'))
        else:
            max_update=max(max(g['measured_update_seconds']) for g in gates)
            remaining=1.5*max_update*p['updates_each']+300
            spent=sum(a['end_epoch']-a['start_epoch'] for a in allocations)
            fit=(time.time()+remaining<window['science_stop_epoch'] and spent+8*remaining<window['GPU_seconds_limit'])
            save(out/'cost_gate.json',dict(passed=fit,update_seconds_max=max_update,conservative_per_task_seconds=remaining,GPU_seconds_used=spent))
            if not fit:raise RuntimeError('full frozen pilot not covered by remaining budget')
            for i in range(8):launch(i,'pilot')
            save(out/'startup.json',dict(started_epoch=time.time(),tasks=p['tasks'],window=window,pilot=True))
            wait_all()
            save(out/'queue_result.json',dict(status='finished',tasks=8,returncodes=[a['returncode'] for a in allocations if a['mode']=='pilot']))
    except BaseException as exc:
        save(out/'supervisor_failure.json',dict(error=repr(exc),epoch=time.time()))
        (out/'STOP').touch()
        for proc,a,log in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        wait_all()
    finally:
        (out/'DONE').touch()
        if watch is not None:watch.wait(timeout=20)
        elapsed=sum((a['end_epoch'] or time.time())-a['start_epoch'] for a in allocations)
        save(out/'resource_receipt.json',dict(wall_seconds=time.time()-t,GPU_hours=elapsed/3600,
                                            all_workers_exited=not running,allocations=allocations))
        subprocess.run([PY,str(HERE/'summarize.py'),str(out)],check=False,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))

if __name__=='__main__':main()
