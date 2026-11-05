"""Conservative admission from measured engineering and historical complete runs."""
import json,os,time,hashlib
from pathlib import Path
def main():
    out=Path(os.environ['STRUCTURE_RUN']);root=Path(__file__).resolve().parents[2]
    read=lambda p:json.loads(p.read_text());ledger=read(out/'ledger.json');receipt=read(out/'engineering/receipt.json');assert receipt['passed']
    old={'IMDB_2026110411_A_late':(70.896,32.244),'IMDB_2026110412_A_late':(74.515,32.952),'PubMed_2026110421_A_late':(92.268,42.775),'PubMed_2026110422_A_late':(94.636,43.163)}
    rows=[]
    for task in read(out/'tasks.json'):
        tid=task['task_id'];r=next(x for x in receipt['rows'] if x['task'].startswith(task['dataset']))
        steady=r['updates'][-1]['update_seconds'];train=max(old[tid][0],steady*3000/60+4)*1.2;chains=old[tid][1]*1.25
        rows.append(dict(task=tid,measured_last_update_seconds=steady,measured_two_reverse_steps=r['reverse_step_seconds'],historical_train_minutes=old[tid][0],historical_four_chain_minutes=old[tid][1],planned_train_minutes=train,planned_four_chain_minutes=chains,device_minutes=train+chains+3))
    now=time.time();eng=sum((x['released_epoch']-x['acquired_epoch']) for p in (out/'leases').glob('*.json') for x in [read(p)])/3600
    hours=eng+sum(x['device_minutes'] for x in rows)/60+.25;finish=now+max(x['device_minutes'] for x in rows)*60
    b=dict(fits=hours<=12 and finish<ledger['science_stop_epoch']-120,epoch=now,tasks=rows,engineering_GPU_hours=eng,planned_GPU_hours=hours,planned_science_finish_epoch=finish,
       chain_estimate='history full chains x1.25; two initial reverse steps are not full-chain throughput',margin='training x1.20; chains x1.25; each card +3min; total +0.25GPUh; CPU delivery reserved30min',ledger=ledger)
    (out/'budget.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b,indent=2))
    roots=[root/'experiments/structure_output_v1',root/'experiments/structure_input_v1',root/'experiments/o_cn_matched_execute_v1',root/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1']
    source={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for base in roots for p in base.rglob('*.py')}
    (out/'source_binding.json').write_text(json.dumps(source,indent=2)+'\n')
if __name__=='__main__':main()
