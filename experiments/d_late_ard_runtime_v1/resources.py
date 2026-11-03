"""Single locked process/lease registry shared by CPU and GPU launchers."""
import fcntl,json,os,time
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'results/d_late_ard_formal_20261103T095019'
def ticks(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
 except OSError:return None
def mutate(fn):
 with (OUT/'registry.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=OUT/'registry.json';r=json.loads(p.read_text());fn(r)
  t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def register(pid,kind,**kw):
 def fn(r):
  if any(p['pid']==pid and p['start_ticks']==ticks(pid) for p in r['processes']):raise ValueError('duplicate process')
  r['processes'].append(dict(pid=pid,start_ticks=ticks(pid),kind=kind,**kw))
 mutate(fn)
def unregister(pid):
 mutate(lambda r:r.update(processes=[p for p in r['processes'] if p['pid']!=pid]))
def lease_start(pid,gpu,uuid,task):
 def fn(r):
  if any(x['uuid']==uuid and not x.get('end_epoch') for x in r['gpu_leases']):raise ValueError('device already leased')
  r['gpu_leases'].append(dict(pid=pid,start_ticks=ticks(pid),gpu=gpu,uuid=uuid,task=task,start_epoch=time.time()))
 mutate(fn)
def lease_end(pid):
 def fn(r):
  for x in r['gpu_leases']:
   if x['pid']==pid and not x.get('end_epoch'):x['end_epoch']=time.time()
 mutate(fn)
