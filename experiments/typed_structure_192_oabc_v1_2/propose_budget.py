"""Read-only 1000-update proposal, using the launcher's exact LPT queues. No admission."""
import argparse,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from admission import assignment
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def proposal(cost_path,resources_path):
    cost=json.loads(pathlib.Path(cost_path).read_text());resources=json.loads(pathlib.Path(resources_path).read_text());samples=resources['gpu_samples'];devices=[];excluded=[]
    parsed=[]
    for sample in samples:
        if sample['returncode']!=0 or sample['compute_returncode']!=0:raise ValueError('resource query failed')
        parsed.append({r.split(',')[1].strip():[x.strip() for x in r.split(',')] for r in sample['output'].splitlines()})
    for uuid,row in parsed[-1].items():
        idle=all(uuid in s and '2080 Ti' in s[uuid][2] and int(s[uuid][4])==0 and int(s[uuid][5])==0 for s in parsed)
        process_absent=all(uuid not in s['compute_processes'] for s in samples)
        if idle and process_absent:devices.append(uuid)
        else:excluded.append(dict(uuid=uuid,index=int(row[0]),reason='not zero utilization/memory in every read-only snapshot, or compute process present'))
    cells={}
    for label,row in cost['cells'].items():
        cells[label]=dict(training_seconds=1000*row['normal2_seconds'],two_full_chains_seconds=2000*row['sampling_step_seconds'],validation_save_seconds=row['validation_save_seconds'],normal2_component_seconds=row['normal2_seconds'],full_step_component_seconds=row['sampling_step_seconds'])
        cells[label]['total_component_seconds']=sum(cells[label][k] for k in ['training_seconds','two_full_chains_seconds','validation_save_seconds'])
    scenarios=[]
    for name,ids in [('current_snapshot',devices),('conditional_eight_idle_cards',list(parsed[-1]))]:
        queues,loads=assignment(cost,1000,ids);per_device=[]
        for uuid,q,load in zip(ids,queues,loads):
            pieces={k:sum(cells[d+'_'+a][k] for d,a in q) for k in ['training_seconds','two_full_chains_seconds','validation_save_seconds']}
            per_device.append(dict(uuid=uuid,queue=[d+'_'+a for d,a in q],**pieces,total_component_seconds=load))
        critical=max(loads);planned=critical*cost['concurrency_margin']+2100
        scenarios.append(dict(name=name,requires_new_idle_check=True,cards=len(ids),per_device=per_device,component_critical_path_seconds=critical,unmeasured_concurrency_allowance_seconds=critical*(cost['concurrency_margin']-1),evaluation_delivery_reserve_seconds=1200,fault_exit_reserve_seconds=900,total_proposed_seconds=planned,total_proposed_minutes=planned/60,proposed_rounded_budget_minutes=int((planned+299)//300)*5))
    result=dict(scope='PROPOSAL ONLY; NOT APPROVAL',updates_per_model=1000,models=8,chains=16,T=1000,source_cost_sha256=sha(cost_path),resource_snapshot_sha256=sha(resources_path),cells=cells,total_task_component_seconds=sum(c['total_component_seconds'] for c in cells.values()),confirmed_idle_at_snapshot=devices,excluded_at_snapshot=excluded,queue_policy='admission.assignment LPT; descending total seconds, stable label ties, next minimum-load device; FIFO per device',scenarios=scenarios,measurement_scope='historical bounded single-GPU component measurements, two GPU processes with four sequential cells each; not 7/8-GPU contention measurement',limits=['25% planning margin is unmeasured, not guaranteed concurrency performance','forward-noise probe states do not bound evolving free-generation CN/NNLS cost','validation/save is component extrapolation with 10 seconds/cell IO allowance, not measured formal throughput','resource snapshot is not a reservation; recheck and recompute identical queue before any separately authorized run'],formal_authorized=False,formal_start=None,formal_deadline=None)
    followup=pathlib.Path(resources_path).parent/'resource_followup.json'
    if followup.exists():
        result['resource_followup']=json.loads(followup.read_text());result['resource_followup_sha256']=sha(followup)
        result['preferred_scenario']=result['resource_followup']['preferred_budget_scenario']
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cost',default=str(ROOT/'evidence/cost_proposal.json'));p.add_argument('--resources',default=str(ROOT/'evidence/current_resources.json'));a=p.parse_args();print(json.dumps(proposal(a.cost,a.resources),indent=2))
