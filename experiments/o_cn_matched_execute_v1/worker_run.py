"""One authorized fresh model, 3000 single-input updates, final EMA two chains."""
import argparse,os,signal,socket,time,traceback,resource,hashlib,pickle
from common_run import ROOT,SCI,OUT,HERE,read,save,sha,now,check_time
import numpy as np,torch
from prepare import load_case
from model_factory import make_model,tensor_sha
from experiment_contract import read_dictionary,reset_training_streams
from probe import rng_get,equal,input_hash
from worker import atomic_checkpoint,validation
from integration import checkpoint_payload,ema_validation,final_readouts
from ema import ParameterEMA
from single_input import one_input,update
from sampling import sample_one
from integration_check import setup
from boundary_transactions import TrainingBoundary
def main():
 p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args();ticket=read(a.ticket);approval=read(OUT/'approval.json');job=ticket['job'];folder=OUT/'run'/job['task_id'];folder.mkdir(exist_ok=False)
 if not approval['authorized'] or os.getppid()!=approval['launcher_pid'] or socket.gethostname()!=approval['host'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(job['gpu_slot']):raise PermissionError('launcher/host/device binding')
 for f,h in approval['source_files'].items():
  if sha(f)!=h:raise ValueError('source/config binding changed '+f)
 status=dict(status='initializing',task_id=job['task_id'],dataset=job['dataset'],train_seed=job['train_seed'],arm=job['arm'],GPU=job['gpu_slot'],GPU_UUID=ticket['GPU_UUID'],pid=os.getpid(),started=now(),completed_updates=0,completed_chains=0)
 completed=chains=0;model=opt=ema=rng=training=None;stopping=False
 def stop(*a):
  nonlocal stopping
  stopping=True
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,stop)
 def check():
  if stopping:raise InterruptedError('own worker stop requested')
  check_time()
 def event(**kw):
  status.update(kw,completed_updates=completed,completed_chains=chains,last_event=now());save(folder/'progress.json',status)
 def snapshot(name):
  atomic_checkpoint(folder/name,training.checkpoint(binding))
 event()
 try:
  setup();check();ds=job['dataset'];seed=job['train_seed'];cfg=job['config'];case,_=load_case(ds)
  basis,_=read_dictionary(SCI/'dictionaries',ds,approval['dictionaries'][ds]) if job['arm']=='E' else (None,None)
  bank=approval['initial_banks'][job['task_id']]
  if sha(bank['file'])!=bank['sha256']:raise ValueError('theta0 bank hash')
  z=torch.load(bank['file'],map_location='cpu')
  if z['updates']!=0 or z['train_seed']!=seed or z['dataset']!=ds or z['arm']!=job['arm'] or z['config']!=cfg:raise ValueError('theta0 identity')
  model=make_model(case,cfg,job['arm']+'_late',basis).cuda();model.load_state_dict(z['model'],strict=True)
  if {k:tensor_sha(v) for k,v in model.state_dict().items()}!=bank['tensor_sha256']:raise ValueError('actual initial tensors')
  opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=ParameterEMA(model)
  if opt.state or {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('new optimizer membership')
  rng,rr=reset_training_streams(seed,ds,'cuda')
  if any(rr['state_sha256'][k]!=v for k,v in z['CPU_training_rng']['state_sha256'].items() if k!='device'):raise ValueError('paired training stream reset')
  def full_rng_hashes():
   return {k:hashlib.sha256(v.cpu().numpy().tobytes() if torch.is_tensor(v) else pickle.dumps(v,protocol=4)).hexdigest() for k,v in rng_get(rng,'cuda').items()}
  if full_rng_hashes()!=bank['expected_initial_rng_sha256']:raise ValueError('sealed A initial full RNG boundary mismatch')
  binding=dict(dataset=ds,train_seed=seed,arm=job['arm'],configuration=job['arm']+'_late',config=cfg,updates=3000,inputs_per_update=1,initial_bank=bank,approval_sha256=sha(OUT/'approval.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'))
  training=TrainingBoundary(model,opt,ema,rng,'cuda',completed)
  save(folder/'binding.json',binding);save(folder/'training_rng_reset.json',rr);snapshot('initial.pt')
  save(folder/'initialization.json',dict(proof=z['initialization'],all_initial_tensors_exact=True,saved_before_first_forward=True,new_optimizer_empty=True,new_EMA_theta0=True,parameters=job['parameters']))
  del z
  from hsgd.bw_preflight.common import environment
  save(folder/'environment.json',environment('cuda'))
  vals=[]
  def validate(u):
   vals.append(dict(updates=u,raw=validation(model,case,cfg,seed,rng),ema_0_99=ema_validation(validation,model,ema,case,cfg,seed,rng)));save(folder/'validation.json',vals)
  validate(0);contracts=read(OUT/'prepared'/f'input_contract_{ds}_{seed}.json');event(status='training')
  for position in range(1,3001):
   check()
   with training.step(position):
    inp=one_input(case,cfg,position,rng);ih=input_hash([inp]);expected=contracts[position-1]
    if ih!=expected['sha256'] or inp['t']!=expected['t'] or len(inp['ids'])!=expected['slots']:raise ValueError('paired input contract before update '+str(position))
    tick=time.perf_counter()
    # Tensor pickle storage identifiers are not stable across processes; hash tensor bytes explicitly.
    def stream_hashes():
     state=rng_get(rng,'cuda')
     return {k:hashlib.sha256(v.cpu().numpy().tobytes() if torch.is_tensor(v) else pickle.dumps(v,protocol=4)).hexdigest() for k,v in state.items()}
    before_streams=stream_hashes();rec=update(model,opt,case,inp,1000,'cuda',ema,completed,transaction=training)
   completed=training.completed
   rec.update(update=completed,t=inp['t'],input_sha256=ih,paired_input_exact=True,training_streams_before=before_streams,training_streams_after=stream_hashes(),at=now(),total_update_seconds=time.perf_counter()-tick)
   import json
   with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   if position%5==0 or position==1:event(status='training',last_loss=rec['loss'],last_update_seconds=rec['total_update_seconds'],input_contract_matched=True)
   if position in [1000,2000,3000]:
    snapshot('final.pt' if position==3000 else f'checkpoint_u{position}.pt');validate(position)
  states=final_readouts(model,ema,completed);cp=sha(folder/'final.pt')
  for kind,sd in [('raw',states['raw']),('ema_0.99',states['ema'])]:
   atomic_checkpoint(folder/f'readout_{kind}.pt',dict(model=sd,readout=kind,decay=.99 if kind!='raw' else None,updates=3000,train_seed=seed,binding=binding,checkpoint_sha256=cp))
  del model,opt,ema,states;model=opt=ema=training=None;torch.cuda.empty_cache()
  row=dict(id=job['task_id'],dataset=ds,train_seed=seed,configuration=job['arm']+'_late',readout_file=str(folder/'readout_ema_0.99.pt'),readout_sha256=sha(folder/'readout_ema_0.99.pt'))
  for ss in [2026102821,2026102822]:
   check();event(status='sampling',sample_seed=ss);sample_one(row,ss,folder/'ema_0.99'/f'sample_{ss}',case,cfg,basis,OUT,check,event);chains+=1
  event(status='complete')
 except BaseException as exc:
  status.update(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
  if training is not None:completed=training.completed;status['recovery']=training.last_failure
  if model is not None and training is not None and training.resumable and training.phase=='idle':
   try:snapshot(f'stopped_u{completed}.pt')
   except BaseException as e:status['checkpoint_save_error']=repr(e)
  else:status.update(incomplete_update_not_presented_as_complete=True,exact_resume_allowed=False,use_previous_complete_checkpoint_only=True)
  event()
 finally:
  status.update(completed_updates=completed,completed_chains=chains,finished=now(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  if torch.cuda.is_initialized():status.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  save(folder/'summary.json',status)
 return 0 if status['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
