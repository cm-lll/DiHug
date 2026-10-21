"""Light parent: deadline begins BEFORE Torch imports/device/data initialization."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_output(path):
    path=Path(path).resolve();protected=[ROOT]
    if any(path==x or path in x.parents or x in path.parents for x in protected):raise ValueError('output overlaps protected package/data/source')
    return path
def plan():
    cfg=read(ROOT/'config.json');m=hashfile(ROOT/'manifest.json')
    value={'config':cfg,'manifest_sha256':m,'effective_updates':72000,'complete_chains':96,'requires_full_target_GPU_preflight':True,'default_start':False}
    value['approval_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest();return value
def supervise(command,out,deadline,receipt,grace=10):
    child=None;reason=None;exitcode=1;began=time.time();stop_epoch=None;received={'signal':None};old={}
    def caught(sig,frame):received['signal']=sig
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[sig]=signal.signal(sig,caught)
    try:
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        print('START',Path(command[1]).name,'output='+str(out),'remaining_seconds='+str(max(0,deadline-time.time())),flush=True)
        # CUDA/HIP/ROCR visibility is inherited unchanged.
        with (out/'launcher.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/'process.json',{'pid':child.pid,'parent_pid':os.getpid(),'command':command,'deadline_epoch':deadline})
            while child.poll() is None:
                if received['signal'] or (out/'STOP').exists() or time.time()>=deadline:
                    reason='signal_'+str(received['signal']) if received['signal'] else ('STOP' if (out/'STOP').exists() else 'timeout')
                    stop_epoch=time.time()
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                    break
                time.sleep(.2)
            exitcode=child.wait()
            # Successful child exit is necessary but never sufficient.
            if reason is not None:exitcode=exitcode or 2
            if exitcode==0:
                result=read(out/receipt)
                good=(result.get('target_GPU_gate_passed') and result.get('full_IMDB_cost_complete') and bool(result.get('checks'))) if receipt=='preflight.json' else result.get('status')=='completed'
                if not good and result.get('status')!='passed_engineering_fixture':raise ValueError('missing complete result evidence')
    except BaseException:
        exitcode=1
        reason=reason or 'exception_or_missing_evidence'
        raise
    finally:
        cleanup=time.time()
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        for sig,fn in old.items():signal.signal(sig,fn)
        if reason is not None and receipt=='preflight.json':
            previous=read(out/receipt) if (out/receipt).exists() else {}
            write(out/receipt,dict(previous,status='incomplete',stop_reason=reason,target_GPU_gate_passed=False,full_IMDB_cost_complete=False,
                checks=previous.get('checks',[]),not_algorithm_failure=reason in ('timeout','STOP') or reason.startswith('signal_')))
        write(out/'supervisor.json',{'status':'completed' if exitcode==0 and reason is None else 'failed_or_incomplete','child_exit_code':exitcode,'reason':reason,
            'wall_seconds':time.time()-began,'deadline_epoch':deadline,'cleanup_extra_seconds':max(0,time.time()-deadline),
            'stop_cleanup_seconds':time.time()-stop_epoch if stop_epoch is not None else 0,
            'all_own_children_reaped':child is None or child.poll() is not None,'SIGKILL_not_catchable':True})
    print('END child_exit='+str(exitcode),'reason='+str(reason),'output='+str(out),flush=True)
    return int(exitcode) if exitcode>=0 else 128-exitcode
def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    pf=s.add_parser('preflight');pf.add_argument('--output',type=Path,required=True);pf.add_argument('--device',choices=['cuda','cpu'],default='cuda');pf.add_argument('--fixture-only',action='store_true')
    fm=s.add_parser('formal');fm.add_argument('--output',type=Path,required=True);fm.add_argument('--preflight',type=Path,required=True);fm.add_argument('--confirmed-budget-seconds',type=int);fm.add_argument('--approved-plan-sha256');fm.add_argument('--persistent-output-confirmed',action='store_true');fm.add_argument('--resume',action='store_true')
    s.add_parser('plan')
    for name in ('progress','stop'):
        q=s.add_parser(name);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='plan':print(json.dumps(plan(),indent=2));return 0
    if a.mode=='progress':
        for n in ('activation.json','process.json','progress.json','storage.json','preflight.json','summary.json','supervisor.json'):
            if (a.output/n).exists():
                value=read(a.output/n)
                if n=='preflight.json':
                    value={k:value.get(k) for k in ('status','error','checks','target_GPU_gate_passed','full_IMDB_cost_complete','wall_seconds','resources')}
                print(n,json.dumps(value))
        return 0
    out=safe_output(a.output)
    if a.mode=='stop':
        if not out.is_dir():raise FileNotFoundError(out)
        (out/'STOP').write_text('explicit safe stop request\n');print('STOP requested; wait for supervisor and last complete checkpoint');return 0
    # Budget begins here, before reading configuration/manifest or spawning Torch.
    activated=time.time();cfg=read(ROOT/'config.json')
    if a.mode=='preflight':
        if out.exists():raise FileExistsError('new independent output required')
        out.mkdir(parents=True);deadline=activated+min(570,cfg['preflight_total_seconds'])
        write(out/'activation.json',{'activation_epoch':activated,'deadline_epoch':deadline,'budget_seconds':deadline-activated,'mode':'preflight','formal_training':False})
        cmd=[sys.executable,str(ROOT/'preflight.py'),'--output',str(out),'--device',a.device,'--deadline',str(deadline)]
        if a.fixture_only:cmd.append('--fixture-only')
        return supervise(cmd,out,deadline,'preflight.json')
    frozen_plan=plan()
    if not a.persistent_output_confirmed:raise ValueError('verify actual persistent volume and pass --persistent-output-confirmed; container path names alone are insufficient')
    protected=a.preflight.resolve()
    if out==protected or out in protected.parents or protected in out.parents:raise ValueError('formal output overlaps protected preflight input')
    if a.resume:
        activation=read(out/'activation.json')
        if activation['plan_sha256']!=frozen_plan['approval_sha256']:raise ValueError('resume plan mismatch')
        deadline=activation['deadline_epoch']
        if time.time()>=deadline:raise TimeoutError('original confirmed budget expired; no reset')
        if (out/'STOP').exists():raise ValueError('STOP still present; explicitly archive STOP before resume')
    else:
        if not a.confirmed_budget_seconds or a.confirmed_budget_seconds<=0 or a.approved_plan_sha256!=frozen_plan['approval_sha256']:
            raise ValueError('formal is disabled by default: user-confirmed total seconds AND exact frozen plan approval hash required')
        if out.exists():raise FileExistsError('fresh-start refuses existing output')
        # Refuse insufficient forecast before making a new formal result dir.
        pf=read(a.preflight/'preflight.json')
        if not pf.get('target_GPU_gate_passed') or not pf.get('full_IMDB_cost_complete'):raise ValueError('complete target GPU preflight required')
        estimate=pf['cost']['forecast']['total_seconds_with_25percent_and_600s_margin']
        if estimate>a.confirmed_budget_seconds*.8:raise ValueError('full experiment forecast exceeds80% of approved budget; request staged decision')
        deadline=activated+a.confirmed_budget_seconds;out.mkdir(parents=True)
        write(out/'activation.json',{'activation_epoch':activated,'deadline_epoch':deadline,'budget_seconds':a.confirmed_budget_seconds,'plan_sha256':frozen_plan['approval_sha256'],'includes_restart_downtime':True})
    cmd=[sys.executable,str(ROOT/'formal.py'),'--output',str(out),'--preflight',str(a.preflight.resolve()),'--deadline',str(deadline)]
    if a.resume:cmd.append('--resume')
    return supervise(cmd,out,deadline,'summary.json',grace=30)
if __name__=='__main__':raise SystemExit(main())
