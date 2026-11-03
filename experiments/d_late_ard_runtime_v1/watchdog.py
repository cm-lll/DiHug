"""Independent per-ticket guard. Only PID+start-time-bound registered tasks."""
import os,json,time,signal
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/'results/d_late_ard_formal_20261103T095019'
def read(p):return json.loads(p.read_text())
def alive(p):
    try:return Path(f'/proc/{p["pid"]}/stat').read_text().split()[21]==p['start_ticks']
    except OSError:return False
def rss(pid):
    try:return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except OSError:return 0
def main():
    ledger=read(HERE/'ledger.json');last=0;signaled={}
    while not (HERE/'TICKET_COMPLETE.json').exists():
        now=time.time()
        try:reg=read(HERE/'registry.json')
        except (OSError,json.JSONDecodeError):time.sleep(1);continue
        procs=[p for p in reg['processes'] if alive(p)]
        memory=sum(rss(p['pid']) for p in procs if p.get('kind','').startswith('cpu_'))
        excessive=[p for p in procs if p.get('kind')=='cpu_fit' and rss(p['pid'])>ledger['memory_per_worker_bytes']]
        gpu=sum(max(0,min(now,x.get('end_epoch',now))-x['start_epoch']) for x in reg.get('gpu_leases',[]))
        cpu_count=sum(p.get('kind')=='cpu_fit' for p in procs)
        gpu_count=len({x.get('uuid',x.get('gpu')) for x in reg.get('gpu_leases',[]) if not x.get('end_epoch')})
        reason=('wall_science_limit' if now>=ledger['science_stop_epoch'] else
                'aggregate_memory_limit' if memory>ledger['max_total_memory_bytes'] else
                'individual_cpu_memory_limit' if excessive else
                'cpu_concurrency_limit' if cpu_count>ledger['max_cpu_workers'] else
                'gpu_concurrency_limit' if gpu_count>ledger['max_gpu_count'] else
                'GPU_hour_limit' if gpu>=ledger['max_gpu_seconds']-90*max(gpu_count,1) else None)
        if now-last>=600 or reason:
            with (HERE/'watchdog.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=now,pids=[p['pid'] for p in procs],RSS_bytes=memory,CPU_fit_processes=cpu_count,GPU_devices=gpu_count,GPU_seconds=gpu,remaining_science_seconds=ledger['science_stop_epoch']-now,reason=reason))+'\n')
            last=now
        if reason:
            (HERE/'STOP_SCIENCE').write_text(reason+'\n')
            for p in procs:
                if p['pid'] not in signaled:
                    try:os.kill(p['pid'],signal.SIGTERM);signaled[p['pid']]=now
                    except ProcessLookupError:pass
                elif now-signaled[p['pid']]>60 and alive(p):
                    try:os.kill(p['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
        if now>=ledger['final_deadline_epoch']:break
        time.sleep(2)
    (HERE/'watchdog_exit.json').write_text(json.dumps(dict(epoch=time.time(),signaled=list(signaled),reason='ticket_complete' if (HERE/'TICKET_COMPLETE.json').exists() else 'deadline'))+'\n')
if __name__=='__main__':main()
