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
        recent=root/'results/pb_degree_formal_20261106T170600/run'/task['task_id']
        history=[json.loads(x) for x in (recent/'training.jsonl').read_text().splitlines()]
        recent_train=sum(x['total_update_seconds'] for x in history)/60
        recent_chains=sum(read(recent/'ema_0.99'/f'sample_{ss}'/'binding.json')['seconds'] for ss in [2026102821,2026102822,2026110231,2026110232])/60
        train=(max(recent_train,measured*3000/60)+3)*1.30
        chains=recent_chains*1.50
        rows.append(dict(task=task['task_id'],measured_boundary_update_seconds=measured,measured_one_reverse_seconds=r['reverse_step_seconds'],train_minutes=train,chains_minutes=chains,device_minutes=train+chains+5))
    now=time.time();eng=sum((x['released_epoch']-x['acquired_epoch']) for p in (out/'leases').glob('*.json') for x in [read(p)])/3600
    hours=eng+sum(x['device_minutes'] for x in rows)/60+.25;finish=now+max(x['device_minutes'] for x in rows)*60
    b=dict(fits=hours<=12 and finish<ledger['science_stop_epoch']-120,epoch=now,tasks=rows,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=finish,chain_estimate='same-model PB four complete chains measured earlier today x1.50; not extrapolated from one early step',margin='max(recent full training, new measured update x3000)+3min, x1.30; chains x1.50; 5min per card, .25GPUh extra; 30min delivery reserve',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b,indent=2))
if __name__=='__main__':main()
