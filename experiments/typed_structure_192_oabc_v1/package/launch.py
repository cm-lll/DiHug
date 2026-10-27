"""Lightweight launch gate. No torch import, no fitting, no implicit authorization."""
import argparse,pathlib,json,time,hashlib,subprocess,sys,os,signal

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def gate(root,ledger_path,approval_path,protocol_path,dictionaries,cost_path):
    ledger=json.loads(pathlib.Path(ledger_path).read_text());approval=json.loads(pathlib.Path(approval_path).read_text());protocol=json.loads(pathlib.Path(protocol_path).read_text())
    if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original budget expired')
    if approval.get('independent_review')!='passed' or approval.get('user_continue_confirmed') is not True:raise PermissionError('review + user continuation missing')
    if approval.get('ledger_sha256')!=sha(ledger_path) or approval.get('protocol_sha256')!=sha(protocol_path):raise ValueError('ledger/protocol differs')
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.py','.json','.npz')}
    source=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if approval.get('source_sha256')!=source:raise ValueError('reviewed source differs')
    for dataset in protocol['datasets']:
        d=pathlib.Path(dictionaries)/('dictionary_'+dataset);r=json.loads((d/'selection.json').read_text())
        if r['status']!='passed' or sha(d/'final_basis.npz')!=r['basis_sha256']:raise RuntimeError('dictionary gate not passed: '+dataset)
    cost=json.loads(pathlib.Path(cost_path).read_text())
    if approval.get('cost_sha256')!=sha(cost_path) or cost.get('complete_eight_cells') is not True:raise RuntimeError('complete eight-cell engineering cost evidence missing')
    if approval.get('updates') not in protocol['allowed_update_tiers']:raise ValueError('update tier')
    if any(not c.get('passed') for c in cost['cells'].values()) or len(cost['cells'])!=8:raise RuntimeError('engineering checks not all passed')
    return ledger,approval,cost

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
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={row.split(',')[0].strip():[x.strip() for x in row.split(',')] for row in visible.splitlines()}
    for d in devices:
        if d not in rows or '2080 Ti' not in rows[d][1] or int(rows[d][2])>10 or int(rows[d][3])>0:raise RuntimeError('selected GPU not confirmed idle RTX2080Ti: '+d)
    # Admission for EXACT reviewed tier; no effect-dependent choice or post-start change.
    tasks=[(d,a) for d in ['IMDB','PubMed'] for a in 'OABC'];loads=[0.]*len(devices)
    for d,a in tasks:
        c=cost['cells'][d+'_'+a];duration=approval['updates']*c['normal2_seconds']+2000*c['sampling_step_seconds']+c['validation_save_seconds'];i=loads.index(min(loads));loads[i]+=duration
    expected=max(loads)*cost['concurrency_margin']+2100
    if expected>ledger['deadline_epoch']-time.time():raise TimeoutError('reviewed tier no longer fits; require explicit recorded pre-start lower tier')
    if args.mode=='gate':print(json.dumps(dict(status='gate_passed_no_execution',expected_seconds=expected)));return 0
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);stop=[False];running={};finished=[];failed=False
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
    try:
        while tasks or running:
            if stop[0] or (out/'STOP').exists() or time.time()>=ledger['deadline_epoch']-1200:failed=True;break
            for device in devices:
                if device not in running and tasks and not failed:
                    dataset,arm=tasks.pop(0);label=dataset+'_'+arm;log=(out/(label+'.log')).open('xb');env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    cmd=[sys.executable,str(root/'worker.py'),'--dataset',dataset,'--arm',arm,'--output',str(out/label),'--dictionary-dir',args.dictionary_dir,'--ledger',args.ledger,'--approval',args.approval,'--protocol',args.protocol]
                    process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[device]=(process,log,label)
            for device,(process,log,label) in list(running.items()):
                rc=process.poll()
                if rc is not None:
                    log.close();finished.append(dict(cell=label,exit_code=rc));del running[device]
                    if rc!=0:failed=True
            if failed:break
            time.sleep(.25)
    finally:
        for process,log,label in running.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        cleanup_deadline=min(time.time()+30,ledger['deadline_epoch'])
        for process,log,label in running.values():
            try:process.wait(timeout=max(.1,cleanup_deadline-time.time()))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close();finished.append(dict(cell=label,exit_code=process.returncode,terminated=True))
        complete=not failed and len(finished)==8 and all(r['exit_code']==0 for r in finished)
        (out/'launcher_summary.json').write_text(json.dumps(dict(status='complete' if complete else 'incomplete',jobs=finished,unstarted=tasks,elapsed_since_activation=time.time()-ledger['start_epoch']),indent=2))
    return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
