"""Two training trajectories, then eight existing-kernel chains; no auto approval."""
import argparse,os,pathlib,json,time,subprocess,signal,sys,shutil
from runtime_contract import ROOT,jobs,source_binding,sha,read,save,approval_gate,output_check,BUDGET,RESERVE,SEEDS,asset

def resources(devices,authority,rows,processes,handles):
    if len(devices)!=2 or len(set(devices))!=2:raise ValueError('frozen default queue requires two distinct GPUs')
    if not authority.get('allocation_check_complete') or authority.get('reservation_conflicts') or not set(devices)<=set(authority.get('allowed_uuids',[])):raise PermissionError('GPU allocation not explicitly verified')
    if handles.strip() or any(x.split(',')[0].strip() in devices for x in processes.splitlines() if x.strip()):raise RuntimeError('selected device has another process')
    selected=[]
    for d in devices:
        r=rows[d]
        if '2080 Ti' not in r[2] or float(r[3])<authority['minimum_free_mib']:raise RuntimeError('GPU model/free-memory gate')
        selected.append(dict(index=r[0],uuid=d,name=r[2],free_mib=float(r[3]),used_mib=float(r[4]),utilization_percent=float(r[5])))
    return selected

def summary_process(out,deadline,binding):
    receipt=dict(status='not_started')
    try:
        if time.time()+2>=deadline:return dict(status='not_completed_insufficient_time')
        if source_binding()[1]!=binding:raise ValueError('runtime/summary source changed')
        with (out/'summary_process.log').open('xb') as log:
            p=subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--run',str(out),'--output',str(out/'unified_evaluation.json'),
                              '--deadline',str(deadline-1)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                              stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.time()-1))
        r=read(out/'unified_evaluation.json');receipt=dict(status=r['status'],exit_code=p.returncode,sha256=sha(out/'unified_evaluation.json'))
        if source_binding()[1]!=binding:raise ValueError('summary source changed during evaluation')
        if (p.returncode==0)!=(r['status']=='complete'):raise ValueError('summary completion inconsistent')
    except BaseException as e:receipt=dict(status='failed_or_incomplete',error=repr(e))
    return receipt

def supervise(out,approval_path,ledger_path,devices,spawn=subprocess.Popen,summarizer=summary_process,poll=.25):
    ledger=read(ledger_path);stop=[False];running={};finished=[];failures=[];pending=jobs()['training'].copy();stage='training'
    handlers={s:signal.signal(s,lambda s,f:stop.__setitem__(0,True)) for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]}
    try:
        while pending or running:
            if stop[0] or (out/'STOP').exists() or time.time()>=ledger['deadline_epoch']-RESERVE:raise InterruptedError('signal/STOP/delivery reserve')
            for slot,device in enumerate(devices):
                if slot in running:continue
                job=next((j for j in pending if j['slot']==slot),None)
                if job is None:continue
                pending.remove(job);log=(out/(job['id']+'.log')).open('xb');rd,wr=os.pipe()
                ticket=dict(parent_pid=os.getpid(),job=job,device=device,approval_sha256=sha(approval_path),ledger_sha256=sha(ledger_path))
                os.write(wr,json.dumps(ticket).encode());os.close(wr)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                try:process=spawn([sys.executable,str(ROOT/'run_worker.py'),'--approval',str(approval_path),'--ledger',str(ledger_path),'--launch-fd',str(rd)],
                                  env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(rd,),start_new_session=True)
                except BaseException:log.close();pending.insert(0,job);raise
                finally:os.close(rd)
                running[slot]=(process,log,job)
            for slot,(p,log,j) in list(running.items()):
                code=p.poll()
                if code is not None:
                    log.close();del running[slot];item=dict(job=j,exit_code=code);finished.append(item)
                    try:item['receipt']=read(out/j['directory']/'summary.json')
                    except Exception as e:item['receipt_error']=repr(e)
                    r=item.get('receipt',{})
                    if code!=0 or r.get('status')!='complete' or (j['mode']=='train' and r.get('completed_updates')!=1000) or (j['mode']=='sample' and r.get('completed_chains')!=1):
                        failures.append(item);raise RuntimeError('worker failed/incomplete '+j['id'])
            if not pending and not running and stage=='training':
                # Both final checkpoints must exist before any independent test chain.
                stage='sampling';pending=jobs()['sampling'].copy()
            save(out/'launcher_progress.json',dict(stage=stage,finished=finished,pending=pending,running=[x[2] for x in running.values()]))
            if pending or running:time.sleep(poll)
    except BaseException as e:failures.append(dict(stage=stage,error=repr(e)))
    finally:
        for p,log,j in running.values():
            try:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        cleanup_deadline=min(time.time()+30,ledger['deadline_epoch'])
        for p,log,j in running.values():
            try:p.wait(timeout=max(.01,cleanup_deadline-time.time()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
            log.close();finished.append(dict(job=j,exit_code=p.returncode,terminated=True))
        all_jobs=jobs()['training']+jobs()['sampling'];done={x['job']['id'] for x in finished}
        receipt=dict(status='incomplete',jobs=finished,unstarted=[j for j in all_jobs if j['id'] not in done],primary_failures=failures)
        save(out/'execution_receipt.json',receipt)
        try:receipt['evaluation']=summarizer(out,ledger['deadline_epoch'],ledger['source_sha256'])
        except BaseException as e:receipt['evaluation']=dict(status='failed',error=repr(e))
        if not failures and len(finished)==10 and receipt['evaluation']['status']=='complete':receipt['status']='complete'
        receipt.update(finished_epoch=time.time(),elapsed_seconds=time.time()-ledger['start_epoch']);save(out/'launcher_summary.json',receipt)
        for s,h in handlers.items():signal.signal(s,h)
    return 0 if receipt['status']=='complete' else 2

def main():
    entered=time.time();p=argparse.ArgumentParser();p.add_argument('mode',choices=['dry-run','formal']);p.add_argument('--output',required=True)
    p.add_argument('--approval');p.add_argument('--cost');p.add_argument('--gpu-uuids',nargs='+');a=p.parse_args()
    if a.mode=='dry-run':print(json.dumps(dict(**jobs(),source_sha256=source_binding()[1],formal_activated=False),indent=2));return 0
    if not a.approval or not a.cost or not a.gpu_uuids:p.error('formal requires approved cost, authorization and two GPU UUIDs')
    approval,cost=approval_gate(a.approval,a.cost,a.output)
    protected=[ROOT]+[pathlib.Path(approval.get('initial_paths',{}).get(str(s),asset(s)['initial_path'])).parent for s in SEEDS]
    out=output_check(a.output,protected);devices=a.gpu_uuids
    if devices!=approval['gpu_uuids']:raise ValueError('approved GPU/queue identity differs')
    visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visibility and (not all(v.startswith('GPU-') for v in visibility.split(',')) or not set(devices)<=set(visibility.split(','))):raise PermissionError('cannot override restricted/ambiguous platform visibility')
    rows={r.split(',')[1].strip():[x.strip() for x in r.split(',')] for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    handles=subprocess.run(['fuser']+['/dev/nvidia'+rows[d][0] for d in devices],capture_output=True,text=True)
    if handles.returncode not in (0,1) or (handles.returncode==1 and handles.stderr.strip()):raise RuntimeError('device owner check unavailable')
    resource=resources(devices,approval['resource_authorization'],rows,processes,handles.stdout)
    existing=out.parent
    while not existing.exists():existing=existing.parent
    if shutil.disk_usage(existing).free<4*2**30:raise RuntimeError('less than4GiB free on actual output filesystem')
    if cost['planned_seconds']>BUDGET-(time.time()-entered):raise TimeoutError('approved queue no longer fits')
    # Formal entry time includes all launch checks, never reset after admission.
    ledger=dict(start_epoch=entered,deadline_epoch=entered+BUDGET,timezone='Asia/Shanghai',budget_seconds=BUDGET,output=str(out),
                source_sha256=source_binding()[1],approval_sha256=sha(a.approval),cost_sha256=sha(a.cost))
    out.mkdir(parents=True);save(out/'activation.json',ledger);save(out/'resources.json',resource);save(out/'queue.json',jobs())
    return supervise(out,pathlib.Path(a.approval),out/'activation.json',devices)
if __name__=='__main__':raise SystemExit(main())
