"""Sequential dataset/time supervisor; all imports/startup inside bounded child."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_large_model_20261021_v1')
def save(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def rss(pid):
    try:return int(next(x.split()[1] for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
    except (StopIteration,FileNotFoundError,ProcessLookupError):return 0
def main():
    ledger=json.loads((OUT/'activation.json').read_text());meta=json.loads((OUT/'input_manifest.json').read_text());rows=[];stop=[None];status='incomplete';p=None
    def signal_stop(s,f):stop[0]=signal.Signals(s).name;(OUT/'STOP').touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,signal_stop)
    def kill(child):
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        child.wait()
    try:
        with (OUT/'launch.json').open('x') as f:json.dump(dict(launch_epoch=time.time(),ledger=ledger),f)
        for item in meta:
            remaining=ledger['deadline_epoch']-time.time();key=f"{item['dataset']}_t{item['t']}"
            if not item['resource_gate_pass']:
                rows.append(dict(key=key,status='not_run_resource_gate',resource_gates=item['resource_gates']));continue
            if remaining<420+720 or stop[0] or (OUT/'STOP').exists():
                rows.append(dict(key=key,status='not_run_budget_or_STOP',remaining_seconds=remaining));continue
            log=(OUT/(key+'.log')).open('w');started=time.monotonic();maxrss=0;reason=None
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'probe.py'),item['dataset'],str(item['t'])],env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('START',key,'cap420s RSS8GiB',flush=True)
            while p.poll() is None:
                used=time.monotonic()-started;maxrss=max(maxrss,rss(p.pid))
                if used>=410:reason='item_compute410s_reserve10s_cleanup_within420s'
                elif maxrss>8*1024**3:reason='item_RSS_8GiB'
                elif stop[0] or (OUT/'STOP').exists() or time.time()>=ledger['deadline_epoch']-720:reason='STOP/deadline_delivery_reserve'
                if reason:kill(p);break
                time.sleep(.15)
            code=p.wait();log.close();folder=OUT/key
            receipt=json.loads((folder/'receipt.json').read_text()) if (folder/'receipt.json').exists() else None
            row=dict(key=key,status='completed' if code==0 and receipt and receipt['status']=='completed' else 'failed_or_incomplete',exit_code=code,supervisor_reason=reason,
                child_wall_seconds=time.monotonic()-started,parent_observed_peak_rss_bytes=maxrss,worker_receipt=receipt)
            rows.append(row);save(OUT/'controller_partial.json',dict(rows=rows,elapsed_seconds=time.time()-ledger['activation_epoch']));print('END',key,row['status'],round(row['child_wall_seconds'],2),flush=True);p=None
        status='completed' if len(rows)==6 and all(r['status']=='completed' for r in rows) else 'incomplete'
    except BaseException as exc:rows.append(dict(status='supervisor_failure',reason=repr(exc)))
    finally:
        if p is not None:kill(p)
        save(OUT/'controller.json',dict(status=status,rows=rows,all_children_reaped=True,elapsed_seconds=time.time()-ledger['activation_epoch'],budget_seconds=3600,no_budget_reset=True))
    return int(status!='completed')
if __name__=='__main__':sys.exit(main())
