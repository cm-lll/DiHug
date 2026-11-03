"""Freeze approved identities and paired initial banks before any model forward."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from common_run import ROOT,SCI,OUT,save,sha,read,now,cfg_for
import numpy as np,torch
from prepare import load_case
from experiment_contract import read_dictionary,initialization,reset_training_streams
from probe import input_hash
from hsgd.bw_preflight.common import cpu_tree,plan_for
from single_input import one_input
import hashlib
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 proposal=read(ROOT/'results/frozen_late_method_audit_20261102T121837/recommended_tasks_PROPOSAL_ONLY.json')
 dictionaries={}
 for ds in ['IMDB','PubMed']:
  d=SCI/'dictionaries'/('dictionary_'+ds);r=read(d/'selection.json')
  dictionaries[ds]=dict(K=r['K'],offsets=r['offsets'],dimension=r['offsets'][-1],files={n:sha(d/n) for n in ['final_basis.npz','selection.json','panel.json','frozen_splits.json']})
 save(OUT/'dictionaries.json',dictionaries);tasks=[];banks={};contracts={};pairing={}
 for job in proposal['tasks']:
  ds=job['dataset'];seed=job['train_seed'];arm=job['arm'];identifier=job['task_id'];case,base=load_case(ds);cfg=cfg_for(base,seed)
  basis,_=read_dictionary(SCI/'dictionaries',ds,dictionaries[ds]) if arm=='E' else (None,None)
  m,_,_,proof=initialization(ds,arm,seed,basis);rng,rr=reset_training_streams(seed,ds,'cpu')
  sd=cpu_tree(m.state_dict());hashes={k:tensor_sha(v) for k,v in sd.items()}
  target=OUT/'prepared'/f'{identifier}_theta0.pt'
  if target.exists():raise FileExistsError(target)
  torch.save(dict(model=sd,updates=0,train_seed=seed,dataset=ds,arm=arm,config=cfg,
                  tensor_sha256=hashes,initialization=proof,CPU_training_rng=rr),target)
  banks[identifier]=dict(file=str(target),sha256=sha(target),tensor_sha256=hashes,parameters=proof['parameters'])
  key=f'{ds}_{seed}';pairing.setdefault(key,{})[arm]=hashes
  if key not in contracts:
   rng,_=reset_training_streams(seed,ds,'cpu');rows=[]
   for u in range(1,3001):
    inp=one_input(case,cfg,u,rng);rows.append(dict(update=u,t=inp['t'],slots=len(inp['ids']),sha256=input_hash([inp])))
   path=OUT/'prepared'/f'input_contract_{key}.json';save(path,rows);contracts[key]=dict(file=str(path),sha256=sha(path))
  tasks.append(dict(job,status='authorized_pending_integration',config=cfg,parameters=proof['parameters'],folder=str(OUT/'run'/identifier)))
  del m,sd
 for key,p in pairing.items():
  shared=set(p['A'])&set(p['E']);bad=[k for k in shared if p['A'][k]!=p['E'][k]]
  if bad:raise ValueError('shared initial tensors differ '+key+str(bad))
  pairing[key]=dict(shared_tensor_count=len(shared),all_shared_exact=True,A_parameters=banks[key+'_A_late']['parameters'],E_parameters=banks[key+'_E_late']['parameters'])
 save(OUT/'initial_banks.json',banks);save(OUT/'input_contracts.json',contracts);save(OUT/'initialization_pairing.json',pairing)
 # Same original sampler calls, no network execution and no graph resampling.
 sampling={}
 for ds in ['IMDB','PubMed']:
  case,_=load_case(ds);S=case['space'].total;families=np.repeat(np.arange(case['n_families']),case['space'].counts)
  for ss in [2026102821,2026102822]:
   gen=np.random.default_rng(ss);uniform=gen.random(S);state=(uniform<case['prior'][families]).astype(np.uint8);steps=[]
   for t in range(1000,0,-1):
    p=plan_for(case,[ss,t,61],1);h=hashlib.sha256();ids,f,q=p.block(0)
    h.update(np.array([0,len(ids)],np.int64).tobytes())
    for a in [ids,f,q]:h.update(np.ascontiguousarray(a).tobytes())
    un=gen.random(S);steps.append(dict(t=t,query_plan_sha256=h.hexdigest(),uniform_sha256=hashlib.sha256(un.tobytes()).hexdigest()))
   sampling[f'{ds}_{ss}']=dict(initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),initial_uniform_sha256=hashlib.sha256(uniform.tobytes()).hexdigest(),steps=steps)
 save(OUT/'prepared/sampling_contracts.json',sampling)
 save(OUT/'tasks.json',tasks)
 save(OUT/'protocol.json',dict(method='E-lateCN',baseline='A-lateCN',updates=3000,inputs_per_update=1,CE='sumCE/S with no half factor',EMA=.99,readout='final3000EMA',sampling_T=1000,sample_seeds=[2026102821,2026102822],primary_metrics=['main_incident4_W1','main_joint_degree_TV','triangle_composition_TV'],original_primary=True,LCC_separate=True,nested_statistics='per graph error; within model samples; across train seeds',no_checkpoint_or_seed_selection=True,failures_and_null_preserved=True,training_count=8,final_graphs=16))
 save(OUT/'prepared/receipt.json',dict(at=now(),complete=True,models=8,input_contracts=4,initial_pairs=pairing,dictionaries=dictionaries,model_forward_calls=0,initial_banks_are_untrained=True))
 print('prepared8 fresh banks;4 paired input contracts;4 complete sampling contracts')
if __name__=='__main__':main()
