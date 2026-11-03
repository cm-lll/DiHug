"""Two datasets, three disposable updates each: 1,2,restore1,replay2."""
from common_run import ROOT,SCI,OUT,read,save,sha,now,check_time
import argparse,time,numpy as np,torch,traceback,hashlib,pickle,pathlib
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
 setup();job=next(j for j in read(OUT/'tasks.json') if j['dataset']==ds and j['arm']=='O');seed=job['train_seed'];cfg=job['config'];case,_=load_case(ds)
 basis=None;banks=read(OUT/'initial_banks.json');z=torch.load(banks[job['task_id']]['file'],map_location='cpu')
 model=make_model(case,cfg,'O_late',None).cuda();model.load_state_dict(z['model'],strict=True)
 opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=ParameterEMA(model)
 if opt.state or {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage')
 rng,rr=reset_training_streams(seed,ds,'cuda');binding=dict(dataset=ds,engineering=True)
 actual_rng={k:hashlib.sha256(v.cpu().numpy().tobytes() if torch.is_tensor(v) else pickle.dumps(v,protocol=4)).hexdigest() for k,v in rng_get(rng,'cuda').items()}
 if actual_rng!=banks[job['task_id']]['expected_initial_rng_sha256']:raise ValueError('actual sealed A initial RNG boundary mismatch')
 def snap(u):return checkpoint_payload(model,opt,ema,u,rng_get(rng,'cuda'),binding)
 atomic_checkpoint(dest/'engineering_initial.pt',snap(0))
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
 # End-to-end ONE-step timings, never a new T1000 chain; discard all next states.
 from hsgd.kernel import MarginalKernel
 from hsgd.runtime import reverse_step
 from hsgd.bw_preflight.common import schedule
 source=next(j for j in read(OUT/'reused_A_jobs.json') if j['dataset']==ds and j['train_seed']==seed)
 sample=pathlib.Path(source['folder'])/'ema_0.99/sample_2026102821'
 before_bench=snap(2);old_mode=model.training;model.eval();generation=[]
 try:
  for filename,expected_t in [('Xt1000.pt',1000),('complete_t0499.pt',499),('complete_t0099.pt',99)]:
   check_time();sourcefile=sample/filename;zstate=torch.load(sourcefile,map_location='cpu')
   if zstate['next_t']!=expected_t or zstate['binding']['dataset']!=ds or zstate['binding']['train_seed']!=seed:raise ValueError('benchmark state time/identity')
   state=np.asarray(zstate['state']);torch.cuda.synchronize();tick=time.perf_counter()
   candidate,rec=reverse_step(model,case,state,MarginalKernel(case['prior'],schedule(1000)),expected_t,1,2026102821,np.random.default_rng(2026102821),'cuda',check_time=check_time,trace=False)
   torch.cuda.synchronize();elapsed=time.perf_counter()-tick
   if not rec['exact_coverage'] or rec['blocks']!=1:raise ValueError('benchmark full query coverage')
   generation.append(dict(t=expected_t,source=str(sourcefile),source_sha256=sha(sourcefile),input_edges=int(state.sum()),seconds=elapsed,covered_slots=rec['covered_slots'],discarded_next_state=True))
   del candidate,zstate
 finally:
  model.train(old_mode);rng_set(before_bench['rng'],rng,'cuda')
 if not equal(before_bench,snap(2)):raise ValueError('generation timing polluted training state')
 save(dest/'validation.json',v)
 result=dict(status='passed',dataset=ds,temporary_optimizer_steps=3,records=records,
   CN_absent_and_739_head=True,validation_state_isolation=isolation,loaded_state_exact=loaded,replayed_state_exact=restoration,
   validation_seconds_raw_and_EMA=validation_seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
   generation_step_records=generation,generation_benchmark_training_state_unchanged=True,
   initial_A_full_rng_exact=True,generation_timing_scope='three independent full R1 reverse steps at saved t1000/499/99 on disposable engineering O weights; no complete generated graph',
   peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,actual_GPU=torch.cuda.get_device_name(0),
   all_engineering_weights_excluded_from_formal=True,finished=now())
 save(dest/'receipt.json',result);print(result['status'],ds,[r['update_seconds'] for r in records])
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(OUT/'engineering/failure.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
