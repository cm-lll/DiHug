"""Fresh admission, frozen panel/limits, no effect-based gate."""
import os,json,time
from pathlib import Path
def main():
    out=Path(os.environ['STRUCTURE_RUN']);root=Path(os.environ['STRUCTURE_ASSETS'])
    read=lambda p:json.loads(p.read_text())
    ledger=read(out/'ledger.json');receipt=read(out/'engineering/receipt.json');assert receipt['passed']
    plan=read(root/'results/closure_context_plan_20261107/budget.json');rows=[]
    for task in read(out/'tasks.json'):
        prior=next(x for x in plan['per_task'] if x['task']==task['task_id'])
        measured=next(x for x in receipt['rows'] if x['task'].startswith(task['dataset']))
        # Only the first update has extra gradient probes; subsequent two are full normal updates.
        seconds=max(x['total_boundary_seconds'] for x in measured['updates'][1:])
        train=max(seconds,prior['predicted_full_update_seconds'])*3000/60*1.20
        chains=prior['four_chains_with_25percent_margin_minutes']
        rows.append(dict(task=task['task_id'],measured_boundary_update_seconds=seconds,
            train_minutes=train,
            chains_minutes=chains,device_minutes=train+chains+7))
    now=time.time();eng=sum((r['released_epoch']-r['acquired_epoch']) for p in (out/'leases').glob('*.json') for r in [read(p)])/3600
    hours=eng+sum(r['device_minutes'] for r in rows)/60+.5
    finish=now+max(r['device_minutes'] for r in rows)*60
    result=dict(fits=eng<=.25 and hours<=7 and finish<ledger['science_stop_epoch']-120,
        epoch=now,tasks=rows,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=finish,
        margin='max(actual full non-probe updates, audited historical update+weight cost)*1.20; old full chains*1.25; +7min/card I/O/validation; +.5GPUh reserve',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

