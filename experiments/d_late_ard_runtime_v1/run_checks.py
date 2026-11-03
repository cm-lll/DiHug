"""Bounded integration runner with the same cumulative GPU ledger."""
import os,subprocess,time
from common_run import *
from resources import register,unregister,lease_start,lease_end
def main():
 register(os.getpid(),'gpu_controller');results=[]
 try:
  card=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
  assert int(card[1])<100 and int(card[2])>10000
  for method in ('D','ard'):
   check_time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(TASKOUT/'temporary'))
   with (TASKOUT/f'check_{method}.log').open('x') as log:
    p=subprocess.Popen(['/opt/di-hug-env/bin/python','-u',str(HERE/'recovery_check.py'),'--method',method],env=env,stdout=log,stderr=subprocess.STDOUT)
    register(p.pid,'gpu_engineering',task=method);lease_start(p.pid,0,card[0],'engineering_'+method)
    try:code=p.wait(timeout=12*60)
    finally:
     if p.poll() is None:
      p.terminate()
      try:p.wait(timeout=30)
      except subprocess.TimeoutExpired:p.kill();p.wait()
     lease_end(p.pid);unregister(p.pid)
   r=read(OUT/'engineer'/method/'receipt.json');results.append(r)
   if code or r['status']!='passed':raise RuntimeError('Recovery gate failed: '+method)
  save(OUT/'recovery_gate.json',dict(status='passed',checks=results,temporary_optimizer_steps=sum(r['temporary_optimizer_steps'] for r in results),finished=now()))
 finally:unregister(os.getpid())
if __name__=='__main__':main()
