"""Lightweight launch gate. No torch import, no fitting, no implicit authorization."""
import argparse,pathlib,json,time,hashlib,subprocess,sys,os,signal
from admission import dictionaries as check_dictionaries,assignment,verify_summary
from runtime_plan import source_binding,validate_protocol,plan,task_id,check_activation,BUDGET_SECONDS

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def resource_check(devices,authorization,rows,compute_output,device_users):
    """Permission is explicit; utilization is telemetry, not an ownership test."""
    if len(devices)!=8 or len(set(devices))!=8:raise PermissionError('this run requires exactly eight distinct cards')
    if authorization.get('basis')!='explicit_user_eight_card_authorization' or set(devices)!=set(authorization.get('allowed_uuids',[])):raise PermissionError('selected cards lack explicit resource authorization')
    if authorization.get('reservation_conflicts') or authorization.get('allocation_check_complete') is not True:raise PermissionError('allocation/reservation check incomplete or conflicting')
    if device_users.strip():raise RuntimeError('selected GPU device has open process handles; no interference')
    if any(line.split(',')[0].strip() in devices for line in compute_output.splitlines() if line.strip()):raise RuntimeError('selected GPU has an existing compute task')
    receipt=[]
    for d in devices:
        if d not in rows or '2080 Ti' not in rows[d][2]:raise RuntimeError('selected RTX2080Ti not present: '+d)
        row=rows[d]
        if float(row[3])<authorization['minimum_free_mib'] or float(row[4])>10:raise RuntimeError('insufficient free memory or existing allocation: '+d)
        receipt.append(dict(uuid=d,index=int(row[0]),name=row[2],free_mib=float(row[3]),used_mib=float(row[4]),utilization_percent=float(row[5])))
    return receipt
def gate(root,ledger_path,approval_path,protocol_path,dictionaries,cost_path):
    approval=json.loads(pathlib.Path(approval_path).read_text());protocol=json.loads(pathlib.Path(protocol_path).read_text())
    if approval.get('independent_review')!='passed' or approval.get('user_continue_confirmed') is not True:raise PermissionError('review + explicit user continuation missing')
    if not approval.get('review_reference') or not approval.get('user_authorization_reference'):raise PermissionError('review and authorization evidence references required')
    if approval.get('main_W1_definition_confirmed')!='type_weighted_incident4':raise PermissionError('incident4 must be confirmed')
    if approval.get('protocol_sha256')!=sha(protocol_path):raise ValueError('protocol differs')
    validate_protocol(protocol)
    files,source=source_binding(root)
    if approval.get('source_sha256')!=source:raise ValueError('reviewed source differs')
    verify_summary(root,approval);check_dictionaries(dictionaries,approval)
    if approval.get('updates')!=1000 or approval.get('budget_seconds')!=BUDGET_SECONDS:raise ValueError('only fixed 1000 updates / 145 minutes supported')
    cost=json.loads(pathlib.Path(cost_path).read_text())
    if approval.get('cost_sha256')!=sha(cost_path):raise ValueError('reviewed cost differs')
    assignment(cost,1000,list(range(8)))
    ledger=None
    if ledger_path is not None:
        ledger=json.loads(pathlib.Path(ledger_path).read_text());check_activation(ledger,approval,approval_path,protocol_path)
    return ledger,approval,cost

def run_summary(root,out,approval,deadline):
    """Best effort on both success and failure; never raises over the primary cause."""
    begin=time.time();receipt=dict(status='not_started',started_epoch=begin)
    try:
        if deadline-begin<=2:return dict(status='skipped_insufficient_delivery_time',remaining_seconds=deadline-begin)
        script=verify_summary(root,approval)
        if source_binding(root)[1]!=approval['source_sha256']:raise ValueError('approved runtime/config binding changed before summary')
        with (out/'summary_process.log').open('xb') as log:
            completed=subprocess.run([sys.executable,str(script),'--run',str(out),'--output',str(out/'unified_evaluation.json'),'--deadline-epoch',str(deadline-1)],stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.time()-1))
        receipt['exit_code']=completed.returncode
        verify_summary(root,approval)
        if source_binding(root)[1]!=approval['source_sha256']:raise ValueError('approved runtime/config binding changed during summary')
        path=out/'unified_evaluation.json'
        if path.exists():
            result=json.loads(path.read_text());receipt.update(output_sha256=sha(path),reported_status=result.get('status'))
            if completed.returncode==0 and result.get('status')=='complete':receipt['status']='complete'
            elif completed.returncode==2 and result.get('status')=='incomplete':receipt['status']='incomplete_summary_saved'
            else:receipt['status']='failed_or_inconsistent_summary'
        else:receipt['status']='failed_no_summary_output'
    except subprocess.TimeoutExpired as exc:receipt.update(status='timeout',error=repr(exc))
    except Exception as exc:receipt.update(status='failed',error=repr(exc))
    receipt['seconds']=time.time()-begin
    return receipt

def final_receipt(out,finished,tasks,failed,failures,approval,ledger,evaluation):
    execution_complete=not failed and len(finished)==8 and all(r['exit_code']==0 for r in finished)
    if execution_complete:
        for row in finished:
            try:
                r=json.loads((out/row['cell']/'summary.json').read_text())
                if r.get('status')!='complete' or r.get('completed_updates')!=approval['updates'] or r.get('completed_chains')!=2:raise ValueError('worker completion contract')
            except Exception as exc:execution_complete=False;failures.append(dict(cell=row['cell'],error=repr(exc),stage='worker_receipt'))
    complete=execution_complete and evaluation['status']=='complete'
    return dict(status='complete' if complete else 'incomplete',execution_complete=execution_complete,jobs=finished,unstarted=tasks,primary_failures=failures,evaluation=evaluation,elapsed_since_activation=time.time()-ledger['start_epoch'])

def main():
    entry_epoch=time.time()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['formal','gate','dry-run']);p.add_argument('--approval');p.add_argument('--protocol',required=True);p.add_argument('--dictionary-dir');p.add_argument('--cost');p.add_argument('--output',required=True);p.add_argument('--gpu-uuids',nargs='+');p.add_argument('--manifest-output');args=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    protocol=json.loads(pathlib.Path(args.protocol).read_text());manifest=plan(protocol,args.output)
    if args.mode=='dry-run':
        # Deliberately before admission, subprocesses, CUDA discovery or activation.
        manifest['source_sha256']=source_binding(root)[1]
        if args.cost:
            cost=json.loads(pathlib.Path(args.cost).read_text());queues,loads=assignment(cost,1000,list(range(8)))
            manifest['cost_queue_matches_execution']=True;manifest['estimated_slot_seconds']=loads
            manifest['estimated_total_seconds_with_reserves']=cost['planned_seconds_before_rounding']
        if args.manifest_output:
            with pathlib.Path(args.manifest_output).open('x') as f:json.dump(manifest,f,indent=2)
        else:print(json.dumps(manifest,indent=2))
        return 0
    for name in ['approval','dictionary_dir','cost','gpu_uuids']:
        if not getattr(args,name):p.error('--'+name.replace('_','-')+' required outside dry-run')
    # Activation time is the entry to the formal launcher; checks consume this same window.
    activated=entry_epoch
    _,approval,cost=gate(root,None,args.approval,args.protocol,args.dictionary_dir,args.cost)
    out=pathlib.Path(args.output).resolve()
    if str(out)!=str(pathlib.Path(approval['output']).resolve()):raise ValueError('output is not approved')
    protected=[root.parent.resolve(),pathlib.Path(args.dictionary_dir).resolve()]
    if any(x==out or x in out.parents or out in x.parents for x in protected):raise ValueError('output overlaps protected source/input')
    if out.exists():raise FileExistsError('new output required; no automatic resume/restart')
    devices=args.gpu_uuids
    visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visibility:
        allowed=visibility.split(',')
        if not all(v.startswith('GPU-') for v in allowed) or not set(devices)<=set(allowed):raise PermissionError('ambiguous/restricted platform visibility; no override')
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={row.split(',')[1].strip():[x.strip() for x in row.split(',')] for row in visible.splitlines()}
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    handles=subprocess.run(['fuser']+['/dev/nvidia'+rows[d][0] for d in devices if d in rows],capture_output=True,text=True)
    if handles.returncode not in (0,1) or (handles.returncode==1 and handles.stderr.strip()):raise RuntimeError('cannot verify selected device handles: '+handles.stderr)
    resource_receipt=resource_check(devices,approval['resource_authorization'],rows,compute,handles.stdout)
    queues,loads=assignment(cost,1000,devices);tasks={device:queue for device,queue in zip(devices,queues)}
    expected=cost['planned_seconds_before_rounding']
    if expected>BUDGET_SECONDS-(time.time()-activated):raise TimeoutError('fixed full panel does not fit; no tier change permitted')
    if args.mode=='gate':print(json.dumps(dict(status='gate_passed_no_execution',expected_seconds=expected)));return 0
    ledger=dict(start_epoch=activated,deadline_epoch=activated+BUDGET_SECONDS,budget_seconds=BUDGET_SECONDS,timezone='Asia/Shanghai',output=str(out),approval_sha256=sha(args.approval),source_sha256=approval['source_sha256'],protocol_sha256=sha(args.protocol),cost_sha256=sha(args.cost))
    out.mkdir(parents=True,exist_ok=False);args.ledger=str(out/'activation.json')
    with pathlib.Path(args.ledger).open('x') as f:json.dump(ledger,f,indent=2)
    return supervise(root,args,ledger,approval,cost,devices,tasks,loads,expected,resource_receipt)


def supervise(root,args,ledger,approval,cost,devices,tasks,loads,expected,resource_receipt,spawn=subprocess.Popen,summarizer=run_summary,poll_seconds=.25):
    """Original fail-stop loop; injection is solely for CPU process tests, not a CLI bypass."""
    out=pathlib.Path(args.output);stop=[False];running={};finished=[];failed=False;failures=[]
    (out/'resource_admission.json').write_text(json.dumps(dict(epoch=time.time(),devices=resource_receipt,authority=approval['resource_authorization'],utilization_is_telemetry_only=True),indent=2))
    (out/'schedule.json').write_text(json.dumps(dict(queues=tasks,loads_seconds=loads,expected_seconds=expected,policy='frozen proposal slot order; one model/device, train then sample seeds sequentially'),indent=2))
    previous_signals={sig:signal.signal(sig,lambda s,f:stop.__setitem__(0,True)) for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]}
    try:
        while any(tasks.values()) or running:
            if stop[0] or (out/'STOP').exists() or time.time()>=ledger['deadline_epoch']-1200:
                failed=True;failures.append(dict(stage='supervision',reason='signal' if stop[0] else 'STOP' if (out/'STOP').exists() else 'delivery_reserve_deadline'));break
            for device in devices:
                if stop[0] or (out/'STOP').exists() or time.time()>=ledger['deadline_epoch']-1200:break
                if device not in running and tasks[device] and not failed:
                    dataset,train_seed,arm=tasks[device].pop(0);label=task_id(dataset,train_seed,arm);log=(out/(label+'.log')).open('xb');env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    read_fd,write_fd=os.pipe();ticket=dict(parent_pid=os.getpid(),dataset=dataset,train_seed=train_seed,arm=arm,task_id=label,device=device,approval_sha256=sha(args.approval),cost_sha256=sha(args.cost),output=str((out/label).resolve()),admitted_epoch=time.time())
                    os.write(write_fd,json.dumps(ticket).encode());os.close(write_fd)
                    cmd=[sys.executable,str(root/'worker.py'),'--dataset',dataset,'--arm',arm,'--train-seed',str(train_seed),'--output',str(out/label),'--dictionary-dir',args.dictionary_dir,'--ledger',args.ledger,'--approval',args.approval,'--protocol',args.protocol,'--cost',args.cost,'--launch-fd',str(read_fd)]
                    try:process=spawn(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(read_fd,))
                    except BaseException as exc:
                        log.close();failures.append(dict(stage='worker_start',cell=label,error=repr(exc)));raise
                    finally:os.close(read_fd)
                    running[device]=(process,log,label)
            for device,(process,log,label) in list(running.items()):
                rc=process.poll()
                if rc is not None:
                    log.close();finished.append(dict(cell=label,exit_code=rc));del running[device]
                    if rc!=0:
                        failed=True;failure=dict(stage='worker',cell=label,exit_code=rc)
                        try:failure['worker_receipt']=json.loads((out/label/'summary.json').read_text())
                        except Exception as exc:failure['receipt_error']=repr(exc)
                        failures.append(failure)
            if failed:break
            time.sleep(poll_seconds)
    except BaseException as exc:
        failed=True;failures.append(dict(stage='launcher',error=repr(exc)))
    finally:
        for process,log,label in running.values():
            try:
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            except Exception as exc:failures.append(dict(stage='cleanup_signal',cell=label,error=repr(exc)));failed=True
        cleanup_deadline=min(time.time()+30,ledger['deadline_epoch'])
        for process,log,label in running.values():
            try:process.wait(timeout=max(.1,cleanup_deadline-time.time()))
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            log.close();finished.append(dict(cell=label,exit_code=process.returncode,terminated=True))
        (out/'execution_receipt.json').write_text(json.dumps(dict(jobs=finished,unstarted=tasks,primary_failures=failures,failed=failed),indent=2))
        try:evaluation=summarizer(root,out,approval,ledger['deadline_epoch'])
        except Exception as exc:evaluation=dict(status='failed',error=repr(exc))
        receipt=final_receipt(out,finished,tasks,failed,failures,approval,ledger,evaluation);complete=receipt['status']=='complete'
        (out/'launcher_summary.json').write_text(json.dumps(receipt,indent=2))
        for sig,handler in previous_signals.items():signal.signal(sig,handler)
    return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
