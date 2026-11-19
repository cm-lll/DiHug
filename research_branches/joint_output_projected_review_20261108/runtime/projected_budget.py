"""Admission accounts spent leases plus all prior and proposed remaining work."""
import json,time,statistics
from pathlib import Path
from .formal_runtime import OUT,read,save
PREVIOUS=[OUT.parent/'joint_output_formal_20261108T021741',OUT.parent/'joint_output_formal_controls_20261108T025000']
def budget():
 now=time.time();ledger=read(OUT/'ledger.json');spent=0.;p95={};chain_cost={};remaining=[]
 for root in PREVIOUS+[OUT]:
  for p in (root/'leases').glob('*.json'):
   x=read(p);spent+=(x.get('released_epoch',now)-x['acquired_epoch'])/3600
 for root in PREVIOUS:
  for p in (root/'run').glob('*/training.jsonl'):
   values=sorted(json.loads(s)['total_update_seconds'] for s in p.read_text().splitlines()[5:]);ds=p.parent.name.split('_')[0]
   if values:p95[ds]=max(p95.get(ds,0),values[int(.95*len(values))])
  for p in (root/'run').glob('*/ema_0.99/sample_*/binding.json'):
   x=read(p);chain_cost[x['dataset']]=max(chain_cost.get(x['dataset'],0),x['seconds'])
 for root in PREVIOUS:
  for t in read(root/'tasks.json'):
   progress=root/'run'/t['task_id']/'progress.json';r=read(progress) if progress.exists() else dict(updates=0,chains=0,status='pending')
   if r['status'] in ['complete','failed_or_incomplete']:continue
   ds=t['dataset'];minutes=(3000-r['updates'])*p95[ds]*1.25/60+(4-r['chains'])*chain_cost[ds]*1.25/60+3
   remaining.append(dict(task=t['task_id'],gpu=t['gpu'],minutes=minutes))
 proposed=[];engineering=read(OUT/'engineering/receipt.json')
 for t in read(OUT/'tasks.json'):
  measured=[r['updates'][1]['total_boundary_seconds'] for r in engineering['rows'] if r['task'].startswith(t['dataset'])]
  update=max(measured+[p95[t['dataset']]])
  minutes=3000*update*1.25/60+4*chain_cost[t['dataset']]*1.25/60+3
  proposed.append(dict(task=t['task_id'],gpu=t['gpu'],minutes=minutes,update_seconds=update))
 used=spent+(sum(x['minutes'] for x in remaining+proposed))/60+.5
 waits={i:sum(x['minutes'] for x in remaining if x['gpu']==i) for i in range(8)}
 finish=now+max(waits[x['gpu']]+x['minutes'] for x in proposed)*60
 fits=engineering['passed'] and used<ledger['GPU_hours_limit'] and finish<ledger['science_stop_epoch']-120
 result=dict(fits=fits,epoch=now,spent_GPU_hours=spent,previous_remaining=remaining,proposed=proposed,total_GPU_hours_bound=used,science_finish_bound_epoch=finish,
  source='actual complete training boundaries and complete1000 chains;25% margin;3min per unfinished/proposed worker I/O;0.5GPUh combined reserve',same_ticket=True)
 save(OUT/'budget.json',result);return result
if __name__=='__main__':print(json.dumps(budget()))
