"""Independent wall/GPU-hour watchdog. Only signals PID/start-time-bound own jobs."""
import os,pathlib,signal,time,traceback
from common_run import OUT,read,save,now
def alive(row):
 try:
  s=pathlib.Path('/proc')/str(row['pid'])/'stat';p=s.read_text().split()
  return p[2]!='Z' and p[21]==row['proc_start_ticks']
 except (FileNotFoundError,ProcessLookupError):return False
def monitor():
 ledger=read(OUT/'ledger.json');seen={};sent={};total=0
 while True:
  t=time.time();active=[];total=ledger['prior_GPU_hours']*3600
  for p in (OUT/'leases').glob('*.json'):
   r=read(p);key=str(p);running=alive(r)
   if not running and key not in seen:seen[key]=t
   end=r.get('released_epoch') or seen.get(key) or t
   total+=max(0,end-r['acquired_epoch'])
   if running:active.append(r)
  stop=t>=ledger['science_stop_epoch']-45 or total>=12*3600-4*60
  if stop:
   if not (OUT/'STOP_SCIENCE').exists():save(OUT/'STOP_SCIENCE',dict(at=now(),reason='wall/GPU-hour ceiling',charged_GPU_hours=total/3600))
   for r in active:
    if r['pid'] not in sent:
     os.kill(r['pid'],signal.SIGTERM);sent[r['pid']]=t
    if t>=ledger['science_stop_epoch'] or t-sent[r['pid']]>=40:
     if alive(r):os.kill(r['pid'],signal.SIGKILL)
  save(OUT/'watchdog_state.json',dict(at=now(),pid=os.getpid(),active=active,
      charged_GPU_hours=total/3600,stop_triggered=stop,science_stop=ledger['science_stop'],deadline=ledger['deadline']))
  if (OUT/'DONE').exists() and not active:break
  if t>=ledger['deadline_epoch']:
   save(OUT/'watchdog_deadline.json',dict(at=now(),active=active,charged_GPU_hours=total/3600))
   for r in active:
    if alive(r):os.kill(r['pid'],signal.SIGKILL)
   c=OUT/'controller.json'
   if c.exists():
    r=read(c)
    if alive(r):os.kill(r['pid'],signal.SIGTERM)
   break
  time.sleep(5)
 save(OUT/'watchdog_exit.json',dict(at=now(),GPU_hours=total/3600,reason='DONE or absolute deadline'))
if __name__=='__main__':monitor()
