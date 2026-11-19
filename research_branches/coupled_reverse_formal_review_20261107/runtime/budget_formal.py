"""Admission from this joint-update check and audited matched PB full chains."""
import json,os,time
from pathlib import Path
def main():
    out=Path(os.environ['STRUCTURE_RUN']);root=Path(__file__).resolve().parents[3]
    read=lambda p:json.loads(p.read_text())
    ledger=read(out/'ledger.json');receipt=read(out/'engineering/receipt.json');assert receipt['passed']
    history=read(root/'results/coupled_reverse_scale_20261107/historical_timing.json')
    controls=read(root/'results/coupled_reverse_scale_20261107/reusable_controls.json')
    prior=read(root/'results/coupled_reverse_scale_20261107/next_budget.json');rows=[]
    for task in read(out/'tasks.json'):
        h=next(x for x in history if x['dataset']==task['dataset'] and x['seed']==task['train_seed'])
        r=next(x for x in receipt['rows'] if x['task'].startswith(task['dataset']))
        measured=max(x['total_boundary_seconds'] for x in r['updates'][1:])
        historical=h['total_recorded_update_seconds']+3000*prior['increment_seconds_per_update_extrapolated'][task['dataset']]
        train=max(historical,measured*3000)/60*1.15
        samples=[x for x in controls if x['dataset']==task['dataset'] and x['seed']==task['train_seed']];assert len(samples)==4
        chains=sum(x['chain_seconds'] for x in samples)/60*1.20
        rows.append(dict(task=task['task_id'],measured_boundary_update_seconds=measured,measured_one_reverse_seconds=r['reverse_step_seconds'],train_minutes=train,chains_minutes=chains,device_minutes=train+chains+5))
    now=time.time()
    eng=sum((x['released_epoch']-x['acquired_epoch']) for p in (out/'leases').glob('*.json') for x in [read(p)])/3600
    hours=eng+sum(x['device_minutes'] for x in rows)/60+.25
    finish=now+max(x['device_minutes'] for x in rows)*60
    result=dict(fits=hours<=ledger['GPU_hours_limit'] and finish<ledger['science_stop_epoch']-120,epoch=now,tasks=rows,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=finish,
        chain_estimate='audited matched PB four complete T1000 chains x1.20; unchanged inference; one timestep not treated as a full chain',
        margin='max(measured joint boundary including diagnostic overhead, PB history + predicted increment) x1.15; chains x1.20; +5min/card load/save/validation; +.25GPUh reserve; final30min CPU delivery',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

