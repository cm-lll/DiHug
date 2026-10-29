"""Recover only hash-identical frozen validation inputs and read sealed X499."""
import pathlib,json,hashlib,sys,time
import numpy as np
B=pathlib.Path(__file__).resolve().parent;ROOT=B.parents[1]
SRC=ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_final_screen_v1'
RUN=ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/run'
sys.path[:0]=[str(SRC),str(SRC/'package')]
from runtime_contract import source_binding,verify_assets
from prepare import load_case
from normal2 import pair_inputs
from role_dictionary import seed32
from probe import input_hash
from hsgd.bw_preflight.common import state_edges,plan_for
import torch
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert time.time()<read(B/'ledger.json')['deadline_epoch']
assert source_binding()[1]==read(RUN/'activation.json')['source_sha256'];verify_assets()
case,cfg=load_case('PubMed');cfg=dict(cfg,seed=2026102912)
out=B/'inputs';out.mkdir(exist_ok=False);receipts=[]
def save(name,t,state,ids,f,query,edges,ef,source,labels=None):
 assert state.shape==(18336,) and np.isin(state,[0,1]).all()
 assert np.array_equal(np.sort(ids),np.arange(18336)) and len(set(ids))==18336
 payload=dict(t=np.int64(t),state=state,ids=ids,f=f,query=query,edges=edges,edge_families=ef,types=case['space'].types,schema=np.array(case['space'].pairs),prior=case['prior'])
 if labels is not None:payload['labels']=labels
 p=out/(name+'.npz');np.savez_compressed(p,**payload);receipts.append(dict(name=name,t=t,path=str(p),sha256=sha(p),source=source,edges=int(state.sum()),slots=len(ids)))
for t in [1,500]:
 private=np.random.default_rng(seed32(cfg['seed'],purpose='fixed_validation_noise',time=t));inp=pair_inputs(case,cfg,10000+t,private,t)[0]
 digest=input_hash([inp]);expected=[]
 for arm in ['C','E']:
  v=read(RUN/f'PubMed_s2026102912_{arm}'/'validation.json')[-1];assert v['updates']==3000
  for kind in ['raw','ema_0.99']:expected.append(next(x['input_sha256'] for x in v[kind]['rows'] if x['t']==t))
 assert all(x==digest for x in expected),'frozen validation input not recovered exactly'
 state=np.zeros(18336,dtype=np.uint8);fam,local=case['space'].encode(inp['edges']);state[case['space'].offsets[fam]+local]=1
 save(f'ordinary_t{t}',t,state,inp['ids'],inp['f'],inp['query'],inp['edges'],inp['edge_families'],dict(kind='frozen ordinary forward-noise validation',original_array_file_persisted=False,recovered_by_original_recipe=True,matching_original_input_sha256=digest,comparisons=4,recipe='worker.validation -> seed32(train_seed,purpose=fixed_validation_noise,time=t), pair_inputs(case,cfg,10000+t,private,t)[0]'),inp['labels'])
for arm in ['C','E']:
 folder=RUN/f'PubMed_s2026102912_{arm}/ema_0.99/sample_2026102821';p=folder/'complete_t0499.pt';receipt=read(p.with_suffix('.sha256.json'));assert sha(p)==receipt['sha256']
 ck=torch.load(p,map_location='cpu');assert ck['next_t']==499 and ck['job']['arm']==arm and ck['job']['sample_seed']==2026102821 and ck['job']['readout']=='ema_0.99'
 model_receipt=read(RUN/f'PubMed_s2026102912_{arm}'/'readout_ema_0.99.sha256.json');assert ck['model_sha256']==model_receipt['sha256']
 steps=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()];step=next(x for x in steps if x['time']==500);assert step['remaining_time']==499
 state=np.asarray(ck['state'],np.uint8);edges,ef=state_edges(case,state);ids,f,query=plan_for(case,[2026102821,499,61],1).block(0)
 save(f'chain_{arm}_X499',499,state,ids,f,query,edges,ef,dict(kind='sealed free-generation state after t500 transition',state_file=str(p),state_file_sha256=receipt['sha256'],payload_next_t=ck['next_t'],preceding_log_time=500,preceding_log_remaining_time=499,model_sha256=ck['model_sha256'],no_chain_replay=True))
frozen=dict(source_sha256=source_binding()[1],inputs=receipts,models={arm:dict(path=str(RUN/f'PubMed_s2026102912_{arm}/readout_ema_0.99.pt'),sha256=read(RUN/f'PubMed_s2026102912_{arm}/readout_ema_0.99.sha256.json')['sha256']) for arm in ['C','E']},query_rule='ordinary exact original plan; chain original next-time plan; each C/E pair shares the same frozen full query',bins=dict(endpoint_total_degree=[0,2,5,10,20,40,'inf'],endpoint_relation_degree=[0,1,2,5,10,20,'inf'],common_neighbors=[0,1,2,5,10,'inf']),group_rule='left-closed/right-open intervals; each endpoint stratified separately; CN all/type-specific; existing and absent slots separate',precision='original FP32 network and original p0 export; float64 original posterior',historical_validation_comparison=dict(atol=2e-6,rtol=1e-5,scope='saved aggregate CE and per-family p0 sums only; no historical slot arrays, no claim of slotwise replication'),max_queries=8,optimizer_steps=0,new_graphs=0,frozen_epoch=time.time())
(B/'frozen_panel.json').write_text(json.dumps(frozen,indent=2));print(json.dumps(receipts,indent=2))
