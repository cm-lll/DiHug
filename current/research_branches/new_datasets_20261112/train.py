"""New-dataset A: original single-input CE, 3000 updates, immutable initial state."""
import os,sys,time,signal,traceback,copy
from pathlib import Path
from support import *
from integration import final_readouts
def main():
 out=Path(sys.argv[1]);task=sys.argv[2];p=read(out/'protocol.json');job=next(j for j in p['jobs'] if j['id']==task);folder=out/task/'CE';folder.mkdir(parents=True,exist_ok=False);start=time.time();boundary=None
 def check():
  if time.time()>=p['science_stop_epoch']:raise TimeoutError('science deadline')
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned budget watchdog')))
 def checkpoint(name):
  z=boundary.checkpoint(binding);tmp=folder/(name+'.tmp');torch.save(z,tmp);tmp.replace(folder/name)
 def progress(phase,**kw):base.write(folder/'progress.json',dict(phase=phase,updates=boundary.completed,at_epoch=time.time(),seconds=time.time()-start,**kw))
 try:
  setup();check()
  for path,h in p['bindings'].items():assert base.sha(path)==h,path
  ds=job['dataset'];seed=job['train_seed'];case=case_for(ds);cfg=job['config'];seed_all(seed)
  m=w.StructuralDenoiser(case,cfg,'A').cuda();opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=s.ParameterEMA(m,.99)
  seed_all(seed+1);rng=np.random.default_rng(np.random.SeedSequence([2026111202,seed]));boundary=s.TrainingBoundary(m,opt,ema,rng,'cuda',0)
  binding=dict(dataset=ds,train_seed=seed,arm='A_late',config=cfg,data=str(PREP/ds/'small_192.npz'),data_sha256=base.sha(PREP/ds/'small_192.npz'),protocol_sha256=base.sha(out/'protocol.json'),training='single-input CE, schema denominator, IID t, no added loss',model_init_seed=seed,global_training_seed=seed+1,input_rng_root=2026111202)
  checkpoint('initial_complete.pt');initial=boundary.checkpoint(binding);loaded=torch.load(folder/'initial_complete.pt',map_location='cpu',weights_only=False);assert s.state_digest(initial)==s.state_digest(loaded);del initial,loaded
  base.write(folder/'binding.json',binding);base.write(folder/'environment.json',w.common.environment('cuda'))
  contracts=read(out/'contracts'/f'{task}.json');fixed=[single.one_input(case,cfg,900000+j,np.random.default_rng(np.random.SeedSequence([2026111204,seed,j])),t) for j,t in enumerate([1,75,250,550,800,1000])];valid=[]
  def evaluate():
   m.eval();rows=[]
   with torch.no_grad():
    for inp in fixed:
     check();logits,prob=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True)
     ce=s.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy();groups=[]
     for r in range(case['n_families']):
      for y in [0,1]:
       mask=(inp['f']==r)&(inp['labels']==y);groups.append(dict(relation=r,label=y,n=int(mask.sum()),CE=float(ce[mask].mean()) if mask.any() else None))
     rows.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=float(ce.sum()/case['space'].total),groups=groups))
   return rows
  def validate():
   before=s.state_digest(boundary.checkpoint(binding))
   with boundary.validation():raw=evaluate();ee=s.ema_validation(lambda _:evaluate(),m,ema,boundary=boundary)
   assert before==s.state_digest(boundary.checkpoint(binding));valid.append(dict(update=boundary.completed,raw=raw,EMA=ee));base.write(folder/'validation.json',valid)
  validate();progress('training')
  for pos in range(1,3001):
   check();tick=time.time()
   with boundary.step(pos):
    inp=single.one_input(case,cfg,pos,rng);ih=s.input_hash([inp]);expected=contracts[pos-1];assert ih==expected['input_hash'] and inp['t']==expected['t'],'frozen input contract'
    rb=s.state_digest(s.rng_capture(rng,'cuda'));rec=single.update(m,opt,case,inp,1000,'cuda',ema,pos-1,boundary);ra=s.state_digest(s.rng_capture(rng,'cuda'));assert rb==ra,'model update consumed training RNG unexpectedly'
   rec.update(update=pos,t=inp['t'],input_hash=ih,RNG_hash=ra,input_contract=True,total_seconds=time.time()-tick)
   with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   if pos==1 or pos%10==0:progress('training',last_loss=rec['loss'],last_seconds=rec['total_seconds'])
   if pos%100==0:checkpoint('latest_complete.pt')
   if pos in [1000,2000,3000]:checkpoint(f'checkpoint_u{pos}.pt');validate()
  checkpoint('final_complete.pt');states=final_readouts(m,ema,3000)
  for name,key in [('raw','raw'),('ema_0.99','ema')]:torch.save(dict(model=states[key],readout=name,updates=3000,train_seed=seed,binding=binding,checkpoint_sha256=base.sha(folder/'final_complete.pt')),folder/f'readout_{name}.pt')
  for path,h in p['bindings'].items():assert base.sha(path)==h,path
  progress('complete');base.write(folder/'receipt.json',dict(status='complete',updates=3000,start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,CUDA_recomputed_update_bitwise_not_certified=True))
 except BaseException:
  error=traceback.format_exc();resumable=False
  if boundary is not None and boundary.resumable and boundary.phase=='idle':
   try:checkpoint('stopped_complete.pt');resumable=True
   except BaseException:error+='\n'+traceback.format_exc()
  base.write(folder/'failure.json',dict(error=error,complete_boundary_saved=resumable,use_previous_complete_state_if_false=True,start_epoch=start,end_epoch=time.time()));raise
if __name__=='__main__':main()
