import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/real192_update_audit_20261111'
def read(n): return json.loads((OUT/n).read_text())
def save(n,x):
    p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
    (OUT/'launch.lock').open('x').close()
    window=read('window.json'); tasks=read('protocol.json')['tasks'];records=[];live={};watch=None
    assert time.time()+900<window['science_stop_epoch'],'insufficient time for whole diagnostic panel'
    smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    devices={int(x.split(',')[0]):[s.strip() for s in x.split(',')] for x in smi.splitlines()}
    assert all('2080 Ti' in devices[i][2] and int(devices[i][3].split()[0])<100 and devices[i][1] not in apps for i in range(4))
    save('resource_before.json',dict(epoch=time.time(),devices=smi,apps=apps))
    def publish(done=False):save('registry.json',dict(tasks=records,done=done,supervisor_pid=os.getpid()))
    try:
        publish()
        log=(OUT/'watchdog.log').open('x');watch=subprocess.Popen([sys.executable,str(HERE/'watchdog.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        for i,t in enumerate(tasks):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=(OUT/f'worker_{i}.log').open('x');tick=time.time()
            p=subprocess.Popen([sys.executable,str(HERE/'worker.py'),str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
            live[i]=p;records.append(dict(task=t['id'],gpu=i,pid=p.pid,start_epoch=tick,end_epoch=None));publish()
        save('launch.json',dict(start_epoch=records[0]['start_epoch'],tasks=records,window=window))
        while live:
            for i,p in list(live.items()):
                if p.poll() is not None:
                    records[i].update(end_epoch=time.time(),returncode=p.returncode);del live[i];publish()
            used=sum((r['end_epoch'] or time.time())-r['start_epoch'] for r in records)
            save('progress.json',dict(epoch=time.time(),GPU_hours=used/3600,live_GPUs=list(live)))
            if time.time()>=window['science_stop_epoch'] or used>=window['GPU_seconds_cap']-60:
                for i,p in list(live.items()):stop(p);records[i].update(end_epoch=time.time(),returncode=p.returncode)
                live.clear();publish()
            time.sleep(2)
    except BaseException as ex:save('failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
    finally:
        for i,p in live.items():stop(p);records[i].update(end_epoch=time.time(),returncode=p.returncode)
        publish(True)
        if watch is not None:
            try:watch.wait(timeout=10)
            except subprocess.TimeoutExpired:stop(watch)
        used=sum(r['end_epoch']-r['start_epoch'] for r in records)
        save('resource_receipt.json',dict(end_epoch=time.time(),GPU_hours=used/3600,tasks=records,
             all_ticket_workers_exited=True,after_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)))
if __name__=='__main__':main()
