"""Detached supervisor; writes allocation and PID before waiting for a worker."""
import os,sys,time,json,signal,subprocess,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/same_input_pubmed_completion_20261110'
def save(name,z):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def event(z):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=time.time(),**z))+'\n');f.flush();os.fsync(f.fileno())
def smi(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=20)
def stop(p):
    if p is not None and p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
    start=time.time();window=dict(start_epoch=start,science_stop_epoch=start+210,delivery_stop_epoch=start+300,GPU_stop_seconds=180)
    save('window.json',window);event(dict(event='supervisor_start',pid=os.getpid()));used=0.;tasks=[];reason=None;p=None;alloc=None;peak=0;exit_apps=None
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('supervisor SIGTERM')))
    try:
        for i in (2,3):
            if time.time()>window['science_stop_epoch'] or used>=160:raise TimeoutError('no remaining GPU science budget')
            status=smi(['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']);apps=smi(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
            cols=[x.strip() for x in next(x for x in status.splitlines() if x.startswith('0,')).split(',')]
            if int(cols[2].split()[0])>=100 or int(cols[3].split()[0])>=5 or cols[1] in apps:raise RuntimeError('GPU0 not idle')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            alloc=time.time();peak=0;entry=dict(index=i,start_epoch=alloc,status='allocating');event(dict(event='allocate',**entry));save('active_allocation.json',entry)
            with (OUT/f'worker_{i}.log').open('x') as log:
                p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                entry.update(pid=p.pid,status='running',process_create_time=psutil.Process(p.pid).create_time());save('active_allocation.json',entry);event(dict(event='worker_started',**entry))
                while p.poll() is None:
                    try:peak=max(peak,sum(x.memory_info().rss for x in [psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)))
                    except psutil.NoSuchProcess:pass
                    save('heartbeat.json',dict(epoch=time.time(),pid=p.pid,index=i,GPU_seconds_including_active=used+time.time()-alloc,science_stop_epoch=window['science_stop_epoch']))
                    if time.time()>window['science_stop_epoch'] or used+time.time()-alloc>170:raise TimeoutError('science/GPU deadline')
                    time.sleep(1)
                p.wait()
            elapsed=time.time()-alloc;used+=elapsed;entry.update(status='exited',seconds=elapsed,end_epoch=time.time(),returncode=p.returncode,peak_RSS_MiB=peak/2**20);tasks.append(entry);save('active_allocation.json',entry);event(dict(event='released',**entry));save('supervisor_progress.json',dict(tasks=tasks,GPU_seconds=used));alloc=None
            if p.returncode:raise RuntimeError('worker error')
            p=None
    except BaseException as exc:
        reason=repr(exc);event(dict(event='exception',error=reason));(OUT/'supervisor_failure.txt').write_text(traceback.format_exc())
    finally:
        stop(p)
        if alloc is not None:
            elapsed=time.time()-alloc;used+=elapsed;entry.update(status='stopped',seconds=elapsed,end_epoch=time.time(),returncode=p.returncode if p else None,peak_RSS_MiB=peak/2**20);tasks.append(entry);save('active_allocation.json',entry);event(dict(event='released_after_failure',**entry))
        try:exit_apps=smi(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
        except BaseException as exc:exit_apps='query failed: '+repr(exc)
        receipt=dict(status='completed' if len(tasks)==2 and not reason else 'partial',start_epoch=start,end_epoch=time.time(),wall_seconds=time.time()-start,GPU_hours=used/3600,tasks=tasks,stop_reason=reason,after_apps=exit_apps,all_workers_exited=True)
        save('supervisor.json',receipt)
    # Finish CPU audit even if the conversational tool session disappears.
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');remaining=window['delivery_stop_epoch']-time.time()-15
    if receipt['status']=='completed' and remaining>15:
        with (OUT/'evaluation.log').open('x') as log:
            q=subprocess.Popen([sys.executable,str(Path(__file__).with_name('evaluate_cpu.py'))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            event(dict(event='CPU_evaluation_started',pid=q.pid))
            try:q.wait(timeout=remaining)
            except subprocess.TimeoutExpired:stop(q)
            save('CPU_receipt.json',dict(returncode=q.returncode,end_epoch=time.time(),GPU_hours=0));event(dict(event='CPU_evaluation_finished',returncode=q.returncode))
    event(dict(event='supervisor_finished',GPU_hours=used/3600));save('exit_receipt.json',dict(epoch=time.time(),all_owned_workers_exited=True,GPU_hours=used/3600))
if __name__=='__main__':main()
