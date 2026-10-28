"""Lightweight launch gate. No torch import, no fitting, no implicit authorization."""
import argparse,pathlib,json,time,hashlib,subprocess,sys,os,signal
from admission import dictionaries as check_dictionaries,assignment,verify_summary

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
    ledger=json.loads(pathlib.Path(ledger_path).read_text());approval=json.loads(pathlib.Path(approval_path).read_text());protocol=json.loads(pathlib.Path(protocol_path).read_text())
    if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original budget expired')
    if approval.get('independent_review')!='passed' or approval.get('user_continue_confirmed') is not True:raise PermissionError('review + user continuation missing')
    if approval.get('main_W1_definition_confirmed')!='type_weighted_incident4':raise PermissionError('main W1 definition must be explicitly confirmed before formal execution')
    if approval.get('ledger_sha256')!=sha(ledger_path) or approval.get('protocol_sha256')!=sha(protocol_path):raise ValueError('ledger/protocol differs')
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.py','.json','.npz')}
    source=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if approval.get('source_sha256')!=source:raise ValueError('reviewed source differs')
    verify_summary(root,approval)
    check_dictionaries(dictionaries,approval)
    for dataset in protocol['datasets']:
        d=pathlib.Path(dictionaries)/('dictionary_'+dataset);r=json.loads((d/'selection.json').read_text())
        if r['status']!='passed' or sha(d/'final_basis.npz')!=r['basis_sha256']:raise RuntimeError('dictionary gate not passed: '+dataset)
    cost=json.loads(pathlib.Path(cost_path).read_text())
    if approval.get('cost_sha256')!=sha(cost_path) or cost.get('complete_eight_cells') is not True:raise RuntimeError('complete eight-cell engineering cost evidence missing')
    if approval.get('updates') not in protocol['allowed_update_tiers']:raise ValueError('update tier')
    if any(not c.get('passed') for c in cost['cells'].values()) or len(cost['cells'])!=8:raise RuntimeError('engineering checks not all passed')
    return ledger,approval,cost

def run_summary(root,out,approval,deadline):
    """Best effort on both success and failure; never raises over the primary cause."""
    begin=time.time();receipt=dict(status='not_started',started_epoch=begin)
    try:
        if deadline-begin<=2:return dict(status='skipped_insufficient_delivery_time',remaining_seconds=deadline-begin)
        script=verify_summary(root,approval)
        with (out/'summary_process.log').open('xb') as log:
            completed=subprocess.run([sys.executable,str(script),'--run',str(out),'--output',str(out/'unified_evaluation.json'),'--deadline-epoch',str(deadline-1)],stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.time()-1))
        receipt['exit_code']=completed.returncode
        verify_summary(root,approval)
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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['formal','gate']);p.add_argument('--ledger',required=True);p.add_argument('--approval',required=True);p.add_argument('--protocol',required=True);p.add_argument('--dictionary-dir',required=True);p.add_argument('--cost',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu-uuids',nargs='+',required=True);args=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    ledger,approval,cost=gate(root,args.ledger,args.approval,args.protocol,args.dictionary_dir,args.cost)
    devices=args.gpu_uuids
    if len(devices)!=len(set(devices)) or not 1<=len(devices)<=8:raise ValueError('1..8 distinct allocated GPU UUIDs')
    # Respect scheduler/platform visibility instead of discovering outside allocations.
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
    # Admission for EXACT reviewed tier; no effect-dependent choice or post-start change.
    queues,loads=assignment(cost,approval['updates'],devices)
    tasks={device:queue for device,queue in zip(devices,queues)}
    expected=max(loads)*cost['concurrency_margin']+2100
    if expected>ledger['deadline_epoch']-time.time():raise TimeoutError('reviewed tier no longer fits; require explicit recorded pre-start lower tier')
    if args.mode=='gate':print(json.dumps(dict(status='gate_passed_no_execution',expected_seconds=expected)));return 0
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);stop=[False];running={};finished=[];failed=False;failures=[]
    (out/'resource_admission.json').write_text(json.dumps(dict(epoch=time.time(),devices=resource_receipt,authority=approval['resource_authorization'],utilization_is_telemetry_only=True),indent=2))
    (out/'schedule.json').write_text(json.dumps(dict(queues=tasks,loads_seconds=loads,expected_seconds=expected,policy='LPT; fixed FIFO per device'),indent=2))
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
    try:
        while any(tasks.values()) or running:
            if stop[0] or (out/'STOP').exists() or time.time()>=ledger['deadline_epoch']-1200:
                failed=True;failures.append(dict(stage='supervision',reason='signal' if stop[0] else 'STOP' if (out/'STOP').exists() else 'delivery_reserve_deadline'));break
            for device in devices:
                if device not in running and tasks[device] and not failed:
                    dataset,arm=tasks[device].pop(0);label=dataset+'_'+arm;log=(out/(label+'.log')).open('xb');env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    read_fd,write_fd=os.pipe();ticket=dict(parent_pid=os.getpid(),dataset=dataset,arm=arm,device=device,approval_sha256=sha(args.approval),cost_sha256=sha(args.cost),output=str((out/label).resolve()),admitted_epoch=time.time())
                    os.write(write_fd,json.dumps(ticket).encode());os.close(write_fd)
                    cmd=[sys.executable,str(root/'worker.py'),'--dataset',dataset,'--arm',arm,'--output',str(out/label),'--dictionary-dir',args.dictionary_dir,'--ledger',args.ledger,'--approval',args.approval,'--protocol',args.protocol,'--cost',args.cost,'--launch-fd',str(read_fd)]
                    try:process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(read_fd,))
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
            time.sleep(.25)
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
        evaluation=run_summary(root,out,approval,ledger['deadline_epoch'])
        receipt=final_receipt(out,finished,tasks,failed,failures,approval,ledger,evaluation);complete=receipt['status']=='complete'
        (out/'launcher_summary.json').write_text(json.dumps(receipt,indent=2))
    return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
