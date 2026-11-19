"""Durable bounded queue: two GPU training workers, four CPU sampling workers."""
from r2_common import *
import subprocess,argparse,traceback
def pool(run,jobs,slots,stage):
 queue=list(jobs);running=[];done=0
 while queue or running:
  for item in list(running):
   process,handle,job,slot=item
   if process.poll() is not None:
    handle.close();running.remove(item);done+=1
    if process.returncode:
     # Do not kill other workers: let their current authorized task finish and preserve evidence.
     atomic(run/'controller_failure.json',{'stage':stage,'job':job,'exit_code':process.returncode,'remaining_queue':queue,'other_running':[x[2] for x in running]})
     for p,h,j,s in running:p.wait();h.close()
     raise RuntimeError('worker failure '+job['id'])
  free=[s for s in slots if s not in [x[3] for x in running]]
  while queue and free:
   slot=free.pop(0);job=queue.pop(0);cmd=list(job['cmd']);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
   if stage=='training':env['CUDA_VISIBLE_DEVICES']=str(slot)
   log=(run/'logs'/(job['id'].replace('/','_')+'.log')).open('w')
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True);running.append((p,log,job,slot))
  atomic(run/'status.json',{'stage':stage,'completed_in_stage':done,'total_in_stage':len(jobs),'queued':len(queue),'running':[{'id':j['id'],'pid':p.pid,'slot':s} for p,h,j,s in running],'epoch':time.time()})
  if running:time.sleep(2)
def main(run):
 run=run.resolve();verify_manifest(run);(run/'logs').mkdir(exist_ok=True)
 # Exclusive creation prevents a second coordinator; no auto stale-lock removal.
 with (run/'controller.lock').open('x') as f:f.write(str(os.getpid()))
 start=time.time();jobs=[]
 for ds in ASSETS:
  for seed in (210,211,212):
   for arm in ARMS:jobs.append({'id':f'train/{ds}/{seed}/{arm}','cmd':[sys.executable,'-u',str(ROOT/'scripts/r2_train.py'),'--run',str(run),'--dataset',ds,'--arm',arm,'--seed',str(seed)]})
 pool(run,jobs,[0,1],'training')
 subprocess.run([sys.executable,str(ROOT/'scripts/r2_accept.py'),'--run',str(run),'--stage','training'],check=True,cwd=ROOT)
 jobs=[]
 for ds in ASSETS:
  for seed in (210,211,212):
   for arm in ARMS:
    for prefix in range(100,108):jobs.append({'id':f'generate/{ds}/{seed}/{arm}/{prefix}','cmd':[sys.executable,'-u',str(ROOT/'scripts/r2_generate.py'),'--run',str(run),'--dataset',ds,'--arm',arm,'--seed',str(seed),'--prefix-seed',str(prefix)]})
 pool(run,jobs,list(range(4)),'generation')
 subprocess.run([sys.executable,str(ROOT/'scripts/r2_accept.py'),'--run',str(run),'--stage','generation'],check=True,cwd=ROOT)
 atomic(run/'status.json',{'stage':'completed','fits':18,'outputs':144,'wall_seconds':time.time()-start})
 atomic(run/'controller_exit.json',{'exit_code':0,'wall_seconds':time.time()-start})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 try:main(a.run)
 except BaseException as e:
  atomic(a.run/'controller_exception.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});atomic(a.run/'controller_exit.json',{'exit_code':1});raise
