"""CPU asset gate, fixed initial banks and exact paired input contracts."""
import copy,hashlib,json,pathlib,pickle,shutil,time
import numpy as np
import torch
from common_run import ROOT,CHECKOUT,SCI,OUT,HERE,read,save,sha,now
from prepare import load_case
from experiment_contract import initialization,reset_training_streams
from model_factory import make_model,tensor_sha
from probe import input_hash,equal
from single_input import one_input
from hsgd.bw_preflight.common import cpu_tree,plan_for

HISTORY=ROOT/'results/o_cn_matched_formal_20261103T204646'
CASE=ROOT/'results/d_late_ard_formal_20261103T095019/generation'

def hashes(state):
 return {k:hashlib.sha256(v.cpu().numpy().tobytes() if torch.is_tensor(v) else pickle.dumps(v,protocol=4)).hexdigest() for k,v in state.items()}

def graph_check(job,ss):
 folder=pathlib.Path(job['folder'])/'ema_0.99'/f'sample_{ss}';b=read(folder/'binding.json')
 for key,value in [('dataset',job['dataset']),('train_seed',job['train_seed']),('configuration',job['arm']+'_late'),('updates',3000),('sample_seed',ss),('readout','EMA0.99'),('R',1),('T',1000)]:
  if b[key]!=value:raise ValueError(('old graph identity',key,b[key],value))
 if sha(folder/'graph.npz')!=b['graph_sha256'] or sha(pathlib.Path(job['folder'])/'readout_ema_0.99.pt')!=b['model_sha256']:raise ValueError('old graph/weight hash')
 return dict(dataset=job['dataset'],train_seed=job['train_seed'],arm=job['arm'],sample_seed=ss,graph=str(folder/'graph.npz'),sha256=b['graph_sha256'],binding=str(folder/'binding.json'))

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 pre=read(OUT/'preregistration.json');jobs=pre['queue'];banks={};reused=[];checks=[];protected=[]
 original_science=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
 (SCI/'package/assets').mkdir(exist_ok=True)
 for ds in ['IMDB','PubMed']:
  source=original_science/'package/assets'/f'{ds}.npz';target=SCI/'package/assets'/source.name
  expected=read(SCI/'package/config.json')['datasets'][ds]['sha256']
  if sha(source)!=expected:raise ValueError('original data hash')
  if not target.exists():shutil.copy2(source,target)
  if sha(target)!=expected:raise ValueError('isolated data hash')
 # Reuse the complete historical two-sample panel without re-evaluation science.
 for j in read(HISTORY/'tasks.json')+read(HISTORY/'reused_A_jobs.json'):
  j=dict(j,panel='historical_two_samples',sample_seeds=[2026102821,2026102822],evaluation_folder=str(HISTORY/'evaluation'/j['task_id']))
  records=[graph_check(j,s)for s in j['sample_seeds']];reused.append(j)
  protected.extend(dict(path=x['graph'],sha256=x['sha256']) for x in records)
 case_job=next(j for j in read(CASE/'tasks.json') if j['train_seed']==2026110321 and j['method']=='A')
 case_job=dict(case_job,panel='posthoc_known_bad_A',evaluation_folder=str(CASE/'evaluation'/case_job['task_id']))
 oldA=pathlib.Path(case_job['folder']);sealed=torch.load(oldA/'initial.pt',map_location='cpu')
 if sha(oldA/'initial.pt')!=read(oldA/'initial.sha256.json')['sha256'] or sealed['updates']!=0 or sealed['ema']['updates']!=0 or sealed['optimizer']['state']:raise ValueError('sealed A initial boundary')
 cfg=case_job['config']
 if cfg!=sealed['binding']['config'] or cfg['inputs_per_update']!=1 or cfg['rounds']!=1 or cfg['optimizer']['lr']!=.0002:raise ValueError('case config')
 originalbank=read(CASE/'initial_banks.json')[case_job['task_id']]
 if sha(originalbank['file'])!=originalbank['sha256']:raise ValueError('A initial bank hash')
 z0=torch.load(originalbank['file'],map_location='cpu')
 if not equal(z0['model'],sealed['model']):raise ValueError('A original bank vs actual initial')
 rng,rr=reset_training_streams(2026110321,'PubMed','cpu')
 if {k:v for k,v in rr['state_sha256'].items() if k!='device'}!={k:v for k,v in hashes(sealed['rng']).items() if k!='device'}:raise ValueError('case initial CPU/main-input RNG differs')
 for ss in pre['sample_seeds']:
  r=graph_check(case_job,ss);protected.append(dict(path=r['graph'],sha256=r['sha256']))
 readout=torch.load(oldA/'readout_ema_0.99.pt',map_location='cpu')
 if readout['updates']!=3000 or readout['readout']!='ema_0.99' or readout['decay']!=.99 or readout['train_seed']!=2026110321:raise ValueError('A final EMA identity')
 reused.append(case_job);protected.extend([dict(path=str(oldA/'initial.pt'),sha256=sha(oldA/'initial.pt')),dict(path=str(oldA/'readout_ema_0.99.pt'),sha256=sha(oldA/'readout_ema_0.99.pt'))])
 # Both new configurations start from the same actual A anchor.
 anchors={};configs={};proofs={}
 for ds,seeds in pre['training_seeds'].items():
  for seed in seeds:
   m,case,base,proof=initialization(ds,'A',seed,None)
   acfg=dict(cfg,seed=seed) # The same normal1/CE scientific settings on both datasets.
   anchors[(ds,seed)]=cpu_tree(m.state_dict());configs[(ds,seed)]=acfg;proofs[(ds,seed)]=proof
   del m
 anchors[('PubMed',2026110321)]=sealed['model'];configs[('PubMed',2026110321)]=cfg
 for job in jobs:
  ds,seed,arm=job['dataset'],job['train_seed'],job['arm'];case,_=load_case(ds);acfg=configs[(ds,seed)]
  if acfg['model']!=read(SCI/'package/config.json')['model'] or acfg['optimizer']!=read(SCI/'package/config.json')['optimizer']:raise ValueError('scientific base model/optimizer differs')
  source=anchors[(ds,seed)];model=make_model(case,acfg,arm+'_late',None);target=model.state_dict();new={}
  for name,t in target.items():
   original=source[name];new[name]=original[:,:739].clone() if name=='joint_head.0.weight' and arm=='O' else original.clone()
   if new[name].shape!=t.shape:raise ValueError('initial migration shape '+name)
  removed=sorted(set(source)-set(new))
  if arm=='O' and (not removed or any(not n.startswith('common_neighbor.') for n in removed)):raise ValueError('CN-only removal')
  model.load_state_dict(new,strict=True)
  if any(tensor_sha(v)!=tensor_sha(new[k])for k,v in model.state_dict().items()):raise ValueError('strict loaded initial bytes')
  rng,rr=reset_training_streams(seed,ds,'cpu');path=OUT/'prepared'/f"{job['task_id']}_theta0.pt"
  proof=dict(common_tensors_exact=True,head_first739_columns_exact=arm=='O',removed=removed,source_A_initial=str(oldA/'initial.pt')if seed==2026110321 else 'new frozen A anchor',
    limitation='Common weights, initial streams and Xt/time/query paired; A executes an additional CN-context dropout call, so full Torch dropout draw trajectories differ by architecture.')
  torch.save(dict(model=new,updates=0,train_seed=seed,dataset=ds,arm=arm,config=acfg,initialization=proof,CPU_training_rng=rr),path)
  expected=hashes(sealed['rng']) if seed==2026110321 else {k:v for k,v in rr['state_sha256'].items()if k!='device'}
  banks[job['task_id']]=dict(file=str(path),sha256=sha(path),tensor_sha256={k:tensor_sha(v)for k,v in new.items()},parameters=sum(p.numel()for p in model.parameters()),expected_initial_rng_sha256=expected)
  job.update(config=acfg,parameters=banks[job['task_id']]['parameters'],folder=str(OUT/'run'/job['task_id']))
  key=f'{ds}_{seed}';contract=OUT/'prepared'/f'input_contract_{key}.json'
  if not contract.exists():
   rng,_=reset_training_streams(seed,ds,'cpu');rows=[]
   for u in range(1,3001):
    inp=one_input(case,acfg,u,rng);rows.append(dict(update=u,t=inp['t'],slots=len(inp['ids']),sha256=input_hash([inp])))
   if seed==2026110321:
    actual=[json.loads(s)for s in (oldA/'training.jsonl').read_text().splitlines()]
    if len(actual)!=3000 or any(r['update']!=a['update'] or r['sha256']!=a['input_sha256'] or r['t']!=a['t'] or r['slots']!=a['slots'] for r,a in zip(rows,actual)):raise ValueError('case actual A main-input pairing fails')
   save(contract,rows)
  checks.append(dict(task_id=job['task_id'],strict_initial_load=True,parameters=job['parameters'],input_contract_updates=3000,known_A_actual_hashes_checked=seed==2026110321))
  del model,new
 sampling={}
 # Freeze complete plans/full-slot uniform hashes before any network execution.
 for ds in ['IMDB','PubMed']:
  case,_=load_case(ds);S=case['space'].total;fam=np.repeat(np.arange(case['n_families']),case['space'].counts)
  for ss in pre['sample_seeds']:
   gen=np.random.default_rng(ss);un=gen.random(S);state=(un<case['prior'][fam]).astype(np.uint8);initial=hashlib.sha256(un.tobytes()).hexdigest();steps=[]
   for t in range(1000,0,-1):
    p=plan_for(case,[ss,t,61],1);h=hashlib.sha256();ids,f,q=p.block(0);h.update(np.array([0,len(ids)],np.int64).tobytes())
    for v in [ids,f,q]:h.update(np.ascontiguousarray(v).tobytes())
    un=gen.random(S);steps.append(dict(t=t,query_plan_sha256=h.hexdigest(),uniform_sha256=hashlib.sha256(un.tobytes()).hexdigest()))
   sampling[f'{ds}_{ss}']=dict(initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),initial_uniform_sha256=initial,steps=steps)
 # Old case's contracts must be identical to the newly frozen sampling plan.
 oldcontract=read(CASE/'prepared/sampling_contracts.json')
 for ss in pre['sample_seeds']:
  if sampling[f'PubMed_{ss}']!=oldcontract[f'PubMed_{ss}']:raise ValueError('case full sampling random contract mismatch')
 save(OUT/'prepared/sampling_contracts.json',sampling);save(OUT/'tasks.json',jobs);save(OUT/'initial_banks.json',banks);save(OUT/'reused_jobs.json',reused)
 save(OUT/'protected_parent_assets.json',protected)
 save(OUT/'prepared/receipt.json',dict(status='passed',at=now(),models=9,new_chains=36,reused_models=9,reused_graphs=20,case_A_initial_and_3000_inputs_exact=True,checks=checks,architecture_dropout_trajectory_limitation=proof['limitation'],GPU_used=False,model_forward_calls=0))
 print('CPU preparation passed:9 new models/36 chains,20 graphs reused,case paired.')

if __name__=='__main__':main()
