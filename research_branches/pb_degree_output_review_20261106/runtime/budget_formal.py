"""Admission only, measured boundary updates plus historical complete-chain costs."""
import json,os,time
from pathlib import Path
def main():
    out=Path(os.environ['STRUCTURE_RUN']);root=Path(__file__).resolve().parents[3]
    read=lambda p:json.loads(p.read_text());ledger=read(out/'ledger.json');receipt=read(out/'engineering/receipt.json');assert receipt['passed']
    old=read(root/'results/pb_degree_preparation_20261106/budget.json');rows=[]
    for task in read(out/'tasks.json'):
        h=next(x for x in old['tasks'] if x['task']==task['task_id'])
        r=next(x for x in receipt['rows'] if x['task'].startswith(task['dataset']))
        measured=max(x['total_boundary_seconds'] for x in r['updates'][1:])
        train=max(h['baseline_training_minutes']+15,measured*3000/60)*1.15
        chains=h['baseline_four_chain_minutes']*1.20
        rows.append(dict(task=task['task_id'],measured_boundary_update_seconds=measured,measured_one_reverse_seconds=r['reverse_step_seconds'],train_minutes=train,chains_minutes=chains,device_minutes=train+chains+5))
    now=time.time();eng=sum((x['released_epoch']-x['acquired_epoch']) for p in (out/'leases').glob('*.json') for x in [read(p)])/3600
    hours=eng+sum(x['device_minutes'] for x in rows)/60+.25;finish=now+max(x['device_minutes'] for x in rows)*60
    b=dict(fits=hours<=12 and finish<ledger['science_stop_epoch']-120,epoch=now,tasks=rows,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=finish,chain_estimate='historical four complete A chains x1.20; one reverse step not treated as full-chain measurement',margin='train x1.15, +15min historical-loss scenario, chain x1.20, 5min per card, .25GPUh extra; 30min delivery reserve',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b,indent=2))
if __name__=='__main__':main()
