"""Build only untrained A/D banks; authenticate and reuse frozen ARD theta0."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import shutil,time
from common_run import *
import generation_adapter as adapter
import torch
from prepare import load_case
from model_factory import make_model,tensor_sha
from hsgd.bw_preflight.common import cpu_tree
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 prior=ROOT/'results/adaptive_role_repair_20261103T032520/generation'
 oldbanks=read(prior/'initial_banks.json');oldtasks=read(prior/'tasks.json')
 dictionaries={k:read(prior/'dictionaries.json')[k] for k in ('baseline','ard')}
 dictionaries.update({k:dict(method=k,dataset='PubMed',role_branch=False,status='not_applicable',frozen=True) for k in ('A','D')})
 for k in dictionaries:adapter.load_dictionary(dictionaries[k])
 case,_=load_case('PubMed');prepared=OUT/'prepared';prepared.mkdir(parents=True,exist_ok=False)
 tasks=[];banks={};pairing=[];inputs={}
 reused=read(ROOT/'results/d_late_ard_plan_20261103/reused_graph_inventory.json')['graphs']
 reuse_rows=[]
 for seed in (2026110221,2026110222,2026110321):
  cfg=adapter.canonical_configuration(seed)
  parent=oldbanks[f'PubMed_{seed}_baseline'];assert sha(parent['file'])==parent['sha256']
  z=torch.load(parent['file'],map_location='cpu');assert z['updates']==0 and z['train_seed']==seed and z['config']==cfg
  for method in ('A','D','baseline','ard'):
   arm=method if method in ('A','D') else 'E';identifier=f'PubMed_{seed}_{method}'
   reuse=method=='baseline' or method=='A' and seed!=2026110321
   record=dictionaries[method]
   if not reuse:
    if method=='ard':banks[identifier]=oldbanks[identifier]
    else:
     torch.manual_seed(seed)
     model=make_model(case,cfg,arm+'_late',None);target=model.state_dict()
     excluded=set(z['model'])-set(target)
     allowed=set(adapter.frozen.ROLE_KEYS)|({'refex_projection.weight'} if arm=='A' else set())
     assert excluded==allowed and set(target)<=set(z['model'])
     for k,v in target.items():
      assert v.shape==z['model'][k].shape and v.dtype==z['model'][k].dtype
      v.copy_(z['model'][k])
     model.load_state_dict(target,strict=True)
     hashes={k:tensor_sha(v) for k,v in model.state_dict().items()}
     assert all(hashes[k]==tensor_sha(z['model'][k]) for k in hashes)
     _,streams=adapter.reset_training(seed,'cpu')
     proof=dict(parent=parent,excluded=sorted(excluded),common_tensors_exact=True,optimizer_inherited=False,updates=0)
     file=prepared/(identifier+'_theta0.pt');params=sum(p.numel() for p in model.parameters())
     torch.save(dict(model=cpu_tree(model.state_dict()),updates=0,train_seed=seed,dataset='PubMed',arm=arm,method=method,
          config=cfg,dictionary_record=record,tensor_sha256=hashes,initialization=proof,CPU_training_rng=streams),file)
     banks[identifier]=dict(file=str(file),sha256=sha(file),tensor_sha256=hashes,parameters=params)
     pairing.append(dict(task_id=identifier,**proof));del model
   folder=adapter.baseline_folder(seed) if method=='baseline' else adapter.BASELINE/'run'/f'PubMed_{seed}_A_late' if reuse else OUT/'run'/identifier
   params=(banks[identifier]['parameters'] if not reuse else 6647441 if arm=='A' else 6816401)
   tasks.append(dict(task_id=identifier,dataset='PubMed',train_seed=seed,method=method,arm=arm,configuration=arm+'_late',
       config=cfg,parameters=params,updates=3000,inputs_per_update=1,EMA_decay=.99,sampling_T=1000,query_R=1,
       sample_seeds=list(adapter.SAMPLE_SEEDS),folder=str(folder),reuse_completed_baseline=reuse))
   if reuse:
    for row in reused:
     if row['train_seed']==seed and row['arm']==arm:
      assert sha(row['graph_path'])==row['graph_sha256'] and sha(row['summary_path'])==row['summary_sha256']
      reuse_rows.append(dict(dataset='PubMed',train_seed=seed,method=method,sample_seed=row['sample_seed'],
           graph=row['graph_path'],graph_sha256=row['graph_sha256'],summary=row['summary_path'],summary_sha256=row['summary_sha256'],strict_match=True))
  cp=prior/'prepared'/f'input_contract_PubMed_{seed}.json';original=read(prior/'input_contracts.json')[f'PubMed_{seed}']
  assert sha(cp)==original['sha256'];shutil.copyfile(cp,prepared/cp.name)
  inputs[f'PubMed_{seed}']=dict(file=str(prepared/cp.name),sha256=sha(cp),count=3000)
 shutil.copyfile(prior/'prepared/sampling_contracts.json',prepared/'sampling_contracts.json')
 assert len(tasks)==12 and sum(not j['reuse_completed_baseline'] for j in tasks)==7 and len(reuse_rows)==20
 for name,value in [('tasks',tasks),('dictionaries',dictionaries),('initial_banks',banks),('input_contracts',inputs),('initialization_pairing',pairing),('reused_samples',reuse_rows)]:save(OUT/(name+'.json'),value)
 save(OUT/'preparation_receipt.json',dict(at=now(),new_models=7,reused_graphs=20,parameters={j['method']:j['parameters'] for j in tasks},forward_calls=0))
if __name__=='__main__':main()
