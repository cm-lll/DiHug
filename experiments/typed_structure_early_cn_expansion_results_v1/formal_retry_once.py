"""One expressly authorized startup retry; retain the first activation clock.

Operational adapter only. Uses the reviewed launcher's admission, resources,
ticket creation and supervisor. No model, optimizer or sampling implementation.
"""
import pathlib,sys,json,time,os,subprocess,shutil,argparse
HERE=pathlib.Path(__file__).resolve().parent
SRC=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1')
sys.path.insert(0,str(SRC))
from runtime_contract import read,save,sha,source_binding,jobs,approval_gate,output_check,RESERVE
from run import resources,supervise
from resource_handles import filter_passive

def validate(approval_path,cost_path):
    a=read(approval_path);old=read(HERE/'run/activation.json')
    required={'environment','user_formal_authorized','independent_review','authorization_reference','source_sha256','protocol_sha256','updates','cost_sha256','budget_seconds','output','gpu_uuids','resource_authorization','asset_bindings_sha256','retry_authorized','retry_adapter_sha256','original_activation_sha256','original_approval_sha256'}
    if required-set(a):raise ValueError('approval missing '+str(sorted(required-set(a))))
    for k in ['python_version','torch_runtime_version','packages','device_name']:
        if k not in a['environment']:raise ValueError('missing environment '+k)
    if a['environment']!=read(HERE/'retry_environment_actual.json')['environment']:raise ValueError('actual environment binding')
    if a['retry_authorized'] is not True or a['retry_adapter_sha256']!=sha(__file__):raise PermissionError('one-retry adapter authorization')
    if a['original_activation_sha256']!=sha(HERE/'run/activation.json') or a['original_approval_sha256']!=sha(HERE/'approval.json'):raise ValueError('first failure evidence changed')
    if a['budget_seconds']!=16800 or old['deadline_epoch']!=old['start_epoch']+16800:raise ValueError('original clock mismatch')
    if read(HERE/'run/launcher_summary.json')['status']!='incomplete':raise ValueError('first invocation not stopped/incomplete')
    for j in jobs()['training']:
        folder=HERE/'run'/j['directory'];r=read(folder/'summary.json')
        if r['completed_updates']!=0 or r['completed_chains']!=0 or (folder/'initial.pt').exists():raise ValueError('not a startup-only retry')
    a,c=approval_gate(approval_path,cost_path,a['output'])
    if c['schedule']!=jobs() or a['gpu_uuids']!=read(HERE/'approval.json')['gpu_uuids']:raise ValueError('queue/GPU change')
    out=output_check(a['output'],[SRC,HERE/'run'])
    # The previous 10-minute contingency covers the already incurred fault.
    # Do not reduce science, the20% concurrency factor, or20min delivery reserve.
    elapsed=time.time()-old['start_epoch'];remaining=old['deadline_epoch']-time.time()
    science=c['body_seconds']*c['concurrency_planning_factor']
    contingency_left=max(0.,c['contingency_seconds']-elapsed)
    required_remaining=science+c['delivery_evaluation_seconds']+contingency_left
    budget=dict(elapsed_seconds=elapsed,remaining_seconds=remaining,science_with_concurrency_seconds=science,delivery_seconds=c['delivery_evaluation_seconds'],original_fault_reserve_seconds=c['contingency_seconds'],fault_reserve_remaining_seconds=contingency_left,required_remaining_seconds=required_remaining,slack_seconds=remaining-required_remaining,deadline_epoch=old['deadline_epoch'])
    if c['concurrency_planning_factor']!=1.2 or c['delivery_evaluation_seconds']!=1200:raise ValueError('planning allowances changed')
    if required_remaining>remaining or science>old['deadline_epoch']-RESERVE-time.time():raise TimeoutError('remaining original budget insufficient '+json.dumps(budget))
    return a,c,old,out,budget

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate-only','formal']);p.add_argument('--approval',required=True);p.add_argument('--cost',required=True);args=p.parse_args()
    a,c,old,out,budget=validate(args.approval,args.cost)
    if args.mode=='validate-only':print(json.dumps(dict(status='passed',model_loaded=False,approval_required_fields_complete=True,budget=budget)));return 0
    if (HERE/'retry_once_claim.json').exists():raise FileExistsError('authorized retry already consumed')
    devices=a['gpu_uuids'];rows={r.split(',')[1].strip():[x.strip() for x in r.split(',')] for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    handles=subprocess.run(['fuser']+['/dev/nvidia'+rows[d][0] for d in devices],capture_output=True,text=True)
    if handles.returncode not in (0,1) or (handles.returncode==1 and handles.stderr.strip()):raise RuntimeError('owner query failed')
    unresolved,passive=filter_passive(handles.stdout,devices);resource=resources(devices,a['resource_authorization'],rows,processes,unresolved)
    if [int(r['index']) for r in resource]!=list(range(2,8)):raise ValueError('not frozen GPU2-7 mapping')
    if shutil.disk_usage(HERE).free<12*2**30:raise RuntimeError('disk admission')
    # Recheck the original clock after resource checks, before any worker exists.
    a,c,old,out,budget=validate(args.approval,args.cost)
    with (HERE/'retry_once_claim.json').open('x') as f:json.dump(dict(attempt=2,claimed_epoch=time.time(),approval_sha256=sha(args.approval),budget=budget),f,indent=2)
    ledger=dict(old,output=str(out),approval_sha256=sha(args.approval),cost_sha256=sha(args.cost),retry_attempt=2,retry_entry_epoch=time.time(),original_activation_sha256=sha(HERE/'run/activation.json'),retry_adapter_sha256=sha(__file__))
    out.mkdir();save(out/'activation.json',ledger);save(out/'resources.json',resource);save(out/'passive_handles.json',passive);save(out/'queue.json',jobs());save(out/'remaining_budget.json',budget)
    return supervise(out,pathlib.Path(args.approval),out/'activation.json',devices)
if __name__=='__main__':raise SystemExit(main())
