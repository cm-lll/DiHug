"""Only resume-count/identity adaptation of the sealed trajectory loop."""
import sys,ast,os,time,argparse,traceback,signal,resource,copy
from common import *
sys.path.insert(0,str(SOURCE))
import run_worker as old
import torch,numpy as np
from prepare import load_case
from model_variants import StructuralDenoiser
from early_cn import install
from experiment_contract import read_dictionary
from runtime_contract import dictionary_expected
from ema import ParameterEMA
from integration import restore_payload
from probe import equal,rng_get,rng_set
def resume3000(job):
 row=read(OUT/'preflight.json')['parents'][job['id']]
 if sha(row['path'])!=row['sha256']:raise ValueError('3000 parent changed')
 x=torch.load(row['path'],map_location='cpu');expected=row['binding']
 case,cfg=load_case('IMDB');cfg=dict(cfg,seed=job['train_seed'])
 if x['binding']!=expected or expected['config']!=cfg or expected['source_sha256']!=SOURCE_HASH:raise ValueError('parent scientific/source compatibility')
 if (x['updates'],x['ema']['updates'],x['ema']['decay'])!=(3000,3000,.99):raise ValueError('resume exact3000 EMA')
 basis,_=read_dictionary(SOURCE/'dictionaries','IMDB',dictionary_expected('IMDB'))
 model=install(StructuralDenoiser(case,cfg,'E',basis)).to('cuda');model.count_aux_weight=expected['lambda_count']
 opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
 ema=ParameterEMA(model);rng=np.random.default_rng(0)
 count=restore_payload(x,model,opt,ema,expected,lambda st:rng_set(st,rng,'cuda'))
 exact={k:equal(v,x[k]) for k,v in [('model',model.state_dict()),('optimizer',opt.state_dict()),('ema',ema.state_dict()),('rng',rng_get(rng,'cuda'))]}
 if count!=3000 or not all(exact.values()):raise ValueError('checkpoint load mismatch')
 proof=dict(parent={k:v for k,v in row.items() if k!='binding'},parent_binding=expected,loaded_updates=count,load_exact=exact,raw_loaded_not_ema=True,optimizer_reset_after_restore=False,ema_updates=ema.updates,parameters=sum(p.numel() for p in model.parameters()))
 return model,opt,ema,case,cfg,rng,proof
def verify6000(run,job):
 if job not in jobs():raise ValueError('noncanonical6000 chain')
 d=pathlib.Path(run)/identity(job['train_seed'],job['arm']);r=read(d/'readouts_u6000.json');p=d/r['files']['ema_0.99']['file']
 if (r['dataset'],r['arm'],r['train_seed'],r['updates'])!=('IMDB',job['arm'],job['train_seed'],6000):raise ValueError('readout identity')
 if sha(p)!=r['files']['ema_0.99']['sha256'] or sha(d/'checkpoint_u6000.pt')!=r['checkpoint_sha256']:raise ValueError('readout/checkpoint hash')
 return p,r
class ResumeAdapt(ast.NodeTransformer):
 def visit_Constant(self,node):
  replacements={500:3000,501:3001,'resumed_u500.pt':'resumed_u3000.pt','next_input501_hashes':'next_input3001_hashes',
   '974e5dca8cbdd05ffddc53fd599ae815828d8b56':'c3f9ed007167484e4d336b2768ea2376b92fcb6e',
   '68a93bd2d6f39a71e6d525dac8647d9435df25ec':'completed_IMDB3000_count_aux_length_20261029T231348'}
  if node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
  return node
def functions():
 tree=ast.parse((SOURCE/'run_worker.py').read_text());g=dict(vars(old))
 selected=[]
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in ('paired_validation','export_readouts','sample','train'):
   selected.append(ResumeAdapt().visit(n) if n.name=='train' else n)
 g.update(ROOT=HERE,source_binding=source_binding,resume_parent=resume3000,verify_readout=verify6000,
  jobs=lambda:dict(training=train_jobs(),sampling=jobs()),generation_checkpoints=lambda d:[6000],PROTOCOL=read(HERE/'protocol.json'))
 exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(SOURCE/'run_worker.py'),'exec'),g)
 return g
def main():
 p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);a=p.parse_args();i=a.slot
 if not 0<=i<4 or os.environ.get('CUDA_VISIBLE_DEVICES')!=DEVICES[i]:raise ValueError('physical device binding')
 ledger=read(OUT/'ledger.json');approval=read(OUT/'approval.json')
 if ledger['source_sha256']!=source_binding()[1] or approval['source_sha256']!=ledger['source_sha256']:raise ValueError('source authorization')
 if not approval['user_authorized'] or approval['updates']!=6000 or approval['new_updates_total']!=12000:raise PermissionError('scope')
 job=train_jobs()[i];d=OUT/'run'/job['directory'];d.mkdir(parents=True,exist_ok=False);stop=[False]
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
 def gate():
  check()
  if stop[0]:raise InterruptedError('own worker signal')
 state=dict(status='restoring3000',job=job,completed_updates=3000,new_updates=0,completed_chains=0,started_epoch=time.time())
 try:
  gate();env=old.configure();save(d/'environment.json',env)
  for k in ['python_version','torch_runtime_version','packages','device_name']:
   if env[k]!=approval['environment'][k]:raise ValueError('original environment mismatch '+k)
  functions()['train'](job,d,approval,dict(output=str(OUT/'run'),approval_sha256=sha(OUT/'approval.json')),gate,state)
  state['status']='complete'
 except BaseException as exc:state.update(status='incomplete',error=repr(exc),traceback=traceback.format_exc())
 finally:
  state.update(finished_epoch=time.time(),new_updates=max(0,state['completed_updates']-3000),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  if torch.cuda.is_initialized():state.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  save(d/'summary.json',state)
 return 0 if state['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
