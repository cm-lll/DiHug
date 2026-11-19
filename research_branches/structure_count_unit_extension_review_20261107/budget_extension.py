"""CPU-only refreshed admission. Deadline confirmation is separate from estimates."""
import os,json,time,statistics,datetime
from pathlib import Path
OUT=Path(os.environ['STRUCTURE_RUN']);ROOT=Path(os.environ['STRUCTURE_ASSETS'])
def read(p):return json.loads(p.read_text())
def main():
    ledger=read(OUT/'ledger.json');parent=Path(ledger['parent_run']);old=read(parent/'budget.json');rows=[]
    for task in read(OUT/'tasks.json'):
        relevant=[r for r in old['tasks'] if r['task'].startswith(task['dataset'])]
        current=[]
        for f in (parent/'run').glob(task['dataset']+'*/training.jsonl'):
            rec=[json.loads(x) for x in f.read_text().splitlines()]
            if rec:current.append(statistics.mean(r['total_update_seconds'] for r in rec))
        # Both arms conservatively costed as UNIT; no benefit from effect or early
        # finish assumptions. Retain original history-based conservative margin.
        train=max(max(r['train_minutes'] for r in relevant),max(current)*3000/60*1.15)
        chains=max(r['chains_minutes'] for r in relevant)
        rows.append(dict(task=task['task_id'],gpu=task['gpu'],train_minutes=train,chains_minutes=chains,device_minutes=train+chains+7,current_mean_update_seconds=max(current)))
    now=time.time();own=sum((r.get('released_epoch',now)-r['acquired_epoch']) for p in (OUT/'leases').glob('*.json') for r in [read(p)])/3600
    hours=sum(r['device_minutes'] for r in rows)/60+.4+own
    finish=now+max(r['device_minutes'] for r in rows)*60
    proposed=datetime.datetime(2026,11,7,19,50,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
    result=dict(at=now,tasks=rows,planned_GPU_hours=hours,parent_reserved_GPU_hours=7,combined_reserved_GPU_hours=7+hours,planned_finish_epoch=finish,
        fits=bool(ledger['deadline_confirmed'] and hours<=ledger['GPU_hours_limit'] and 7+hours<=13.5 and finish<ledger['science_stop_epoch']-120),
        fits_proposed_1950=hours<=6.5 and finish<proposed-120,
        margin='max(original validated conservative history budget, current complete update mean*1.15);4chains*1.25;+7min/card load-validation-save;+0.4 GPUh reserve; no occupied GPU during CPU preparation',
        original_deadline_feasible=finish<1794048546-120,deadline_confirmed=ledger['deadline_confirmed'])
    (OUT/'budget.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
