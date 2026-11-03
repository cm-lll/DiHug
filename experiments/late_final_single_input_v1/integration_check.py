"""Two datasets, three disposable updates each: 1,2,restore1,replay2."""
from common_run import ROOT,SCI,OUT,read,save,sha,now,check_time
import argparse,time,numpy as np,torch,traceback
from model_factory import make_model,tensor_sha
from prepare import load_case
from experiment_contract import read_dictionary,reset_training_streams
from probe import rng_get,rng_set,equal,input_hash,metric
from hsgd.computational import FrozenTypedContext
from worker import atomic_checkpoint,validation
from ema import ParameterEMA,cpu_copy
from integration import checkpoint_payload,ema_validation
from single_input import one_input,update
def setup():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if torch.cuda.device_count()!=1 or '2080 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('one allowed2080Ti')
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();ds=a.dataset;dest=OUT/'engineering'/ds;dest.mkdir(exist_ok=False)
 setup();job=next(j for j in read(OUT/'tasks.json') if j['dataset']==ds and j['arm']=='E');seed=job['train_seed'];cfg=job['config'];case,_=load_case(ds)
 basis,_=read_dictionary(SCI/'dictionaries',ds,read(OUT/'dictionaries.json')[ds]);banks=read(OUT/'initial_banks.json');z=torch.load(banks[job['task_id']]['file'],map_location='cpu')
 model=make_model(case,cfg,'E_late',basis).cuda();model.load_state_dict(z['model'],strict=True)
 opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=ParameterEMA(model)
 if opt.state or {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage')
 rng,rr=reset_training_streams(seed,ds,'cuda');binding=dict(dataset=ds,engineering=True)
 def snap(u):return checkpoint_payload(model,opt,ema,u,rng_get(rng,'cuda'),binding)
 atomic_checkpoint(dest/'engineering_initial.pt',snap(0))
 # Fresh A/E shared tensors and zero structural projections yield same theta0 prediction.
 aj=next(j for j in read(OUT/'tasks.json') if j['dataset']==ds and j['train_seed']==seed and j['arm']=='A')
 az=torch.load(banks[aj['task_id']]['file'],map_location='cpu');am=make_model(case,cfg,'A_late',None).cuda();am.load_state_dict(az['model'],strict=True)
 st=rng_get(rng,'cuda');pr=np.random.default_rng(123);inp=one_input(case,cfg,77,pr,500);outs=[]
 for m in [am,model]:
  m.eval();ctx=FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],'integration_zero')
  try:
   with torch.no_grad():outs.append(ctx.predict(inp['query'],inp['f'],.5,'integration_zero',details=True)[1].detach().cpu())
  finally:ctx.close()
 zero=metric(outs[0],outs[1]);del am,az,outs;torch.cuda.empty_cache();rng_set(st,rng,'cuda')
 if not zero['passes']:raise ValueError('fresh paired prediction mismatch')
 records=[];states={};contracts=read(OUT/'prepared'/f'input_contract_{ds}_{seed}.json')
 for pos in [1,2]:
  check_time();inp=one_input(case,cfg,pos,rng)
  if input_hash([inp])!=contracts[pos-1]['sha256']:raise ValueError('input contract')
  r=update(model,opt,case,inp,1000,'cuda',ema,pos-1);records.append(r);states[pos]=snap(pos);atomic_checkpoint(dest/f'engineering_u{pos}.pt',states[pos])
 raw=cpu_copy(model.state_dict());rng_before=rng_get(rng,'cuda');opt_before=cpu_copy(opt.state_dict());ema_before=ema.state_dict();tick=time.perf_counter()
 v=dict(raw=validation(model,case,cfg,seed,rng),ema=ema_validation(validation,model,ema,case,cfg,seed,rng));validation_seconds=time.perf_counter()-tick
 isolation=equal(raw,cpu_copy(model.state_dict())) and equal(rng_before,rng_get(rng,'cuda')) and equal(opt_before,cpu_copy(opt.state_dict())) and equal(ema_before,ema.state_dict())
 if not isolation:raise ValueError('validation changed train state')
 z1=torch.load(dest/'engineering_u1.pt',map_location='cpu');model.load_state_dict(z1['model'],strict=True);opt.load_state_dict(z1['optimizer']);ema.load_state_dict(z1['ema'],model,1);rng_set(z1['rng'],rng,'cuda')
 loaded=equal(z1,snap(1))
 if not loaded:raise ValueError('checkpoint not exactly loaded')
 inp=one_input(case,cfg,2,rng);records.append(update(model,opt,case,inp,1000,'cuda',ema,1));replayed=snap(2)
 restoration={k:equal(states[2][k],replayed[k]) for k in ['model','optimizer','ema','rng','updates']}
 if not all(restoration.values()):raise ValueError('same-GPU replay changed completed state')
 save(dest/'validation.json',v)
 result=dict(status='passed',dataset=ds,temporary_optimizer_steps=3,records=records,
   zero_projection_pair=zero,validation_state_isolation=isolation,loaded_state_exact=loaded,replayed_state_exact=restoration,
   validation_seconds_raw_and_EMA=validation_seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
   peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,actual_GPU=torch.cuda.get_device_name(0),
   all_engineering_weights_excluded_from_formal=True,finished=now())
 save(dest/'receipt.json',result);print(result['status'],ds,[r['update_seconds'] for r in records])
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(OUT/'engineering/failure.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
