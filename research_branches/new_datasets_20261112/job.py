"""One owned GPU reservation: CE, baseline chains, terminal adaptation/chains."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=Path(sys.argv[1]).resolve();task=sys.argv[2];p=read(out/'protocol.json');j=next(x for x in p['jobs'] if x['id']==task);folder=out/task;folder.mkdir(exist_ok=False)
 here=Path(__file__).parent
 def run(script,args,log):
  with (folder/log).open('w') as f:subprocess.run([sys.executable,str(here/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
 run('train.py',[out,task],'CE.log')
 readout=folder/'CE/readout_ema_0.99.pt';h=hashlib.sha256(readout.read_bytes()).hexdigest();ds=j['dataset'];q=dict(p)
 q.update(parents={ds:dict(readout=str(readout),data=j['data'],train_seed=j['train_seed'])},terminal_plan={ds:j['terminal_plan']},bindings={**p['bindings'],str(readout):h})
 save(folder/'protocol.json',q)
 run('terminal_worker.py',[folder,ds,'baseline'],'baseline.log')
 run('terminal_worker.py',[folder,ds,'T_eager'],'terminal.log')
 save(folder/'complete.json',dict(status='complete',at_epoch=time.time(),CE_updates=3000,terminal_updates=4,evaluation_graphs=8,training_graphs=8))
if __name__=='__main__':main()
