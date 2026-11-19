"""Calibrated queue admission; no quality outcome used to pick/skip arms."""
import os,json,time
from pathlib import Path
def main():
    out=Path(os.environ['STRUCTURE_RUN']);root=Path(os.environ['STRUCTURE_ASSETS']);read=lambda p:json.loads(p.read_text())
    ledger=read(out/'ledger.json');receipt=read(out/'engineering/receipt.json');assert receipt['passed']
    plan=read(root/'results/closure_context_plan_20261107/budget.json');rows=[];devices={i:0. for i in range(8)}
    tasks=read(out/'tasks.json')
    for task in tasks:
        prior=next(x for x in plan['per_task'] if x['dataset']==task['dataset'] and x['seed']==task['train_seed'])
        same=[x for x in receipt['rows'] if next(t for t in tasks if t['task_id']==x['task'])['dataset']==task['dataset'] and next(t for t in tasks if t['task_id']==x['task'])['architecture']==task['architecture'] and next(t for t in tasks if t['task_id']==x['task'])['objective_kind']==task['objective_kind']]
        assert len(same)==1
        seconds=max(x['total_boundary_seconds'] for x in same[0]['updates'][1:])
        train=max(seconds,prior['predicted_full_update_seconds'])*3000/60*1.35
        chains=prior['four_chains_with_25percent_margin_minutes']*1.2
        duration=train+chains+10
        rows.append(dict(task=task['task_id'],gpu=task['gpu'],measured_boundary_update_seconds=seconds,train_minutes=train,chains_minutes=chains,device_minutes=duration,queue_start_minutes=devices[task['gpu']]))
        devices[task['gpu']]+=duration
    now=time.time();eng=sum((r['released_epoch']-r['acquired_epoch']) for p in (out/'leases').glob('*.json') for r in [read(p)]) /3600
    hours=eng+sum(x['device_minutes'] for x in rows)/60+1
    queue=max(devices.values())
    result=dict(fits=eng<=ledger['engineering_GPU_hours_limit'] and hours<=ledger['GPU_hours_limit'] and now+queue*60<ledger['science_stop_epoch']-120,
        epoch=now,tasks=rows,queue_minutes=queue,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=now+queue*60,
        margin='max(actual full updates, historical)*1.35; historical full chains including prior1.25 margin*1.2;10min/card load validation I/O;1GPUh reserve',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

