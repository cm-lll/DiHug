"""Three disposable updates: save1, continuous2, reload1 and repeat2."""
import argparse,os,time,traceback
from common_run import *
import generation_adapter as adapter
import torch
from prepare import load_case
from model_factory import make_model,tensor_sha
from probe import rng_get,rng_set,equal,input_hash
from worker import atomic_checkpoint,validation
from integration import checkpoint_payload,restore_payload,ema_validation
from ema import ParameterEMA
from single_input import one_input,update
from generation_worker import setup
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['D','ard'],required=True);a=p.parse_args()
 folder=OUT/'engineer'/a.method;folder.mkdir(parents=True,exist_ok=False)
 start=time.time();count=0;result=dict(status='started',method=a.method,started=now(),temporary_optimizer_steps=0)
 try:
  setup();check_time();cfg=adapter.canonical_configuration(2026110221);case,_=load_case('PubMed')
  record=read(OUT/'dictionaries.json')[a.method];basis,dictionary=adapter.load_dictionary(record)
  adapter.install_dictionary_encoder(record,basis)
  bank=read(OUT/'initial_banks.json')[f'PubMed_2026110221_{a.method}'];assert sha(bank['file'])==bank['sha256']
  z=torch.load(bank['file'],map_location='cpu');arm='D' if a.method=='D' else 'E';binding=dict(engineering=True,bank_sha256=bank['sha256'],method=a.method)
  def objects():
   m=make_model(case,cfg,arm+'_late',basis).cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=ParameterEMA(m);r,_=adapter.reset_training(2026110221,'cuda');return m,o,e,r
  model,opt,ema,rng=objects();model.load_state_dict(z['model'],strict=True);ema=ParameterEMA(model)
  assert {k:tensor_sha(v) for k,v in model.state_dict().items()}==bank['tensor_sha256']
  def state(u):return checkpoint_payload(model,opt,ema,u,rng_get(rng,'cuda'),binding)
  first=one_input(case,cfg,1,rng);update1=update(model,opt,case,first,1000,'cuda',ema,0);count+=1
  saved=state(1);atomic_checkpoint(folder/'engineering_u1.pt',saved)
  # Fixed validation must leave every training state, mode and RNG intact.
  mode=model.training
  validation(model,case,cfg,2026110221,rng);ema_validation(validation,model,ema,case,cfg,2026110221,rng)
  validation_exact=equal(state(1),saved) and model.training==mode
  assert validation_exact,'validation changed full training state or model mode'
  inp=one_input(case,cfg,2,rng);ih=input_hash([inp]);before=rng_get(rng,'cuda')
  update2=update(model,opt,case,inp,1000,'cuda',ema,1);count+=1;continuous=state(2)
  atomic_checkpoint(folder/'continuous_u2.pt',continuous)
  del model,opt,ema;torch.cuda.empty_cache()
  model,opt,ema,rng=objects();loaded=torch.load(folder/'engineering_u1.pt',map_location='cpu')
  assert sha(folder/'engineering_u1.pt')==read(folder/'engineering_u1.sha256.json')['sha256']
  # This engineering fixture was saved above immediately after successful update1,
  # before drawing input2, and its bytes are authenticated. Never use this opt-in
  # for a legacy abnormal-stop snapshot merely because EMA counters match.
  assert restore_payload(loaded,model,opt,ema,binding,lambda s:rng_set(s,rng,'cuda'),legacy_complete_boundary_verified=True)==1
  assert equal(state(1),saved),'restored checkpoint differs'
  inp2=one_input(case,cfg,2,rng);assert ih==input_hash([inp2]) and equal(before,rng_get(rng,'cuda'))
  replay=update(model,opt,case,inp2,1000,'cuda',ema,1);count+=1;actual=state(2)
  comparisons={k:equal(continuous[k],actual[k]) for k in ('model','optimizer','ema','rng','updates','binding')}
  atomic_checkpoint(folder/'reloaded_u2.pt',actual)
  finite=all(torch.isfinite(p).all() for p in model.parameters()) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  result.update(comparisons=comparisons,validation_preserves_all_state=validation_exact,input_sha256=ih,
    next_input_rng_exact=True,finite=bool(finite),update_records=[update1,update2,replay],
    parameter_max_abs_difference=max(float((continuous['model'][k]-actual['model'][k]).abs().max()) for k in actual['model']),
    formal_initial_unchanged=sha(bank['file'])==bank['sha256'],no_generated_graphs=True)
  assert all(comparisons.values()) and finite,'same-next-input continuation differs'
  result['status']='passed'
 except BaseException as exc:
  result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  result.update(temporary_optimizer_steps=count,seconds=time.time()-start,finished=now())
  if torch.cuda.is_initialized():result.update(peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
  save(folder/'receipt.json',result)
if __name__=='__main__':main()
