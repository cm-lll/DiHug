"""Continue registered temporary trajectories from exact saved first-step states.
Record every fixed t500 response; no new objectives or reverse generation.
"""
import os,sys,json,time,traceback,signal,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(os.environ['OBJECTIVE_DIAG_OUT']);S=R/'results/direct_structure_response_20261108T205816';FIRST=R/'results/direct_one_step_20261108T212756'
spec=importlib.util.spec_from_file_location('helper',R/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(R/'research_branches/reproducible_aggregation_20261108'));from adapter import execution_policy,VERSION
sys.path.insert(0,str(R/'research_branches/direct_structure_cpu_20261108'));from score import DirectScore
P=w.read(O/'protocol.json');started=time.time();stop=False;status=dict(status='running',forwards=0,backwards=0,updates=0,completed=[]);updates=[];probes=w.read(O/'probes.json');checks=[]
def halt(*a):
 global stop
 stop=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
 if stop or time.time()>=P['science_stop_epoch']-20:raise TimeoutError('ticket stop')
def save(n,x):fr.save(O/n,x)
def main():
 fr.setup();fr.authenticate_sources();assert torch.cuda.is_available() and torch.cuda.device_count()==1 and w.update.__globals__['LAMBDA']==1.
 for path,h in w.read(O/'source_binding.json').items():assert w.sha(path)==h
 for path,h in w.read(O/'asset_binding.json').items():assert w.sha(path)==h
 prior_updates=w.read(S/'updates.json')+w.read(S/'control_updates.json');meta=w.read(S/'control_binding.json')
 for task in w.read(R/'results/output_boundary_replay_20261108T081654/tasks.json'):
  ds,cfg=task['dataset'],task['row']['config'];data=R/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  fixed=next(x[2] for x in torch.load(S/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs'] if x[0]==500)
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);rng=np.random.default_rng(1);b=None
  def predict_stats(prob,inp):
   q=inp['query'];y=inp['labels'];n=len(case['space'].types);A=np.zeros((n,n));A[q[0],q[1]]=A[q[1],q[0]]=prob;Y=np.zeros_like(A);Y[q[0],q[1]]=Y[q[1],q[0]]=y
   return dict(expected_edges=float(prob.sum()),expected_triangles=float((A*(A@A)).sum()/6),reference_triangles=float((Y*(Y@Y)).sum()/6),expected_degree_MAE=float(abs(A.sum(1)-Y.sum(1)).mean()))
  def evaluate(arm,offset):
   check();before=g.digest(b.checkpoint(binding))
   with b.validation():
    m.eval();token=('step_trace',condition,arm,offset);ctx=g.FrozenTypedContext(m,case['space'].types,fixed['edges'],fixed['edge_families'],token)
    try:
     with torch.no_grad():
      graph,logits,p=ctx.predict(fixed['query'],fixed['f'],.5,token,details=True);status['forwards']+=1;g.common.check_prob_device(p,fixed['f'],case,m.prior);ce=w.supported_binary_ce(logits,torch.as_tensor(fixed['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy();prob=p.cpu().double().numpy()
     assert np.isfinite(prob).all() and np.isfinite(ce).all();y=fixed['labels'];rels=[]
     for f in range(case['n_families']):
      pos=(fixed['f']==f)&(y==1);neg=(fixed['f']==f)&(y==0);ix=fixed['f']==f
      rels.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),expected_edges=float(prob[ix].sum()),missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None))
     probes.append(dict(condition=condition,seed=task['seed'],arm=arm,offset=offset,updates=b.completed,t=500,input_sha256=w.input_hash([fixed]),CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[y==1].mean()),negative_CE=float(ce[y==0].mean()),relations=rels,**predict_stats(prob,fixed)));save('probes.json',probes)
     np.savez_compressed(O/f'{condition}_{arm}_offset{offset}.npz',probability=prob,CE=ce,query=fixed['query'],families=fixed['f'],labels=y,types=case['space'].types)
     if offset==8:
      with np.load(S/f'{condition}_{arm}_t500.npz') as z:assert np.array_equal(prob,z['probability']) and np.array_equal(ce,z['CE']),'final probe differs from original eight steps'
    finally:ctx.close()
   assert before==g.digest(b.checkpoint(binding)),'evaluation changes state/RNG'
  class Objective:
   def __init__(self,arm,aux):self.arm=arm;self.aux=aux;self.last={}
   def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
   def __call__(self,p,q,f,inp,*,logits):
    prob=p.detach().cpu().double().numpy();self.last=dict(training_prediction=predict_stats(prob,inp))
    np.savez_compressed(O/f'{condition}_{self.arm}_train_offset{offset}.npz',probability=prob,query=q,families=f,labels=inp['labels'],types=case['space'].types,t=inp['t'])
    if self.arm=='CE':return p.sum()*0,p.new_empty(0)
    before=g.digest(g.rng_capture(rng,'cuda'));bits=(self.aux.random(len(prob))<prob).astype(float);direct=DirectScore(case['space'].types,q,f,inp['labels'],bits,case['n_families']);gradient=direct.logit_estimate(prob);assert np.isfinite(gradient).all();margin=logits[:,1]-logits[:,0];linear=(margin*torch.as_tensor(gradient,device='cuda',dtype=margin.dtype)).sum();aux=P['coefficient']*(linear-linear.detach()+direct.base)
    np.savez_compressed(O/f'{condition}_gradient_offset{offset}.npz',gradient=gradient,bits=bits)
    self.last.update(hard_score_sample=direct.base,aux_random_state=self.aux.bit_generator.state,margin_gradient_L2=float(np.linalg.norm(gradient)))
    assert before==g.digest(g.rng_capture(rng,'cuda'))
    return aux,p.new_tensor([direct.base])
  with execution_policy(True):
   save('environment.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for parentarm in ['CE','PS']:
    condition=ds+'_'+parentarm;ends={}
    for arm in ['CE','CE_STRUCT']:
     check();cp=FIRST/f'{condition}_{arm}_temporary.pt';env=torch.load(cp,map_location='cpu');parent=env['state'];binding=parent['binding'];assert env['parent_sha256']==meta[condition]['parent_sha256'] and binding['config']==cfg and parent['updates']==task['start']+33
     g.restore_payload(w.cpu_copy(parent),m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();o.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,o,e,rng,'cuda',parent['updates']);restored=b.checkpoint(binding)
     for k in ['model','optimizer','ema','rng','updates']:assert g.digest(restored[k])==g.digest(parent[k]),'restore '+k
     aux=np.random.default_rng(P['aux_seeds'][ds])
     if arm=='CE_STRUCT':
      with np.load(FIRST/f'{condition}_CE_STRUCT_before_training.npz') as z:aux.random(len(z['probability']))
      first=next(x for x in prior_updates if x['condition']==condition and x['arm']==arm and x['offset']==1);assert aux.bit_generator.state==first['aux_random_state'],'auxiliary boundary mismatch'
     obj=Objective(arm,aux)
     for offset in range(2,9):
      check();u=b.completed+1;before={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
      with b.step(u):
       inp=g.one_input(case,cfg,u,rng);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256'];assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_before']
       rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=obj);status['updates']+=1;status['forwards']+=1;status['backwards']+=1;assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_after']
      delta=torch.cat([(p.detach().cpu()-before[n]).flatten() for n,p in m.named_parameters()]);move=w.stats(delta);old=next(x for x in prior_updates if x['condition']==condition and x['arm']==arm and x['offset']==offset)
      for k in ['unweighted_CE','gradient_norm','auxiliary_loss']:assert rec[k]==old[k],f'historical {k} mismatch at {condition}/{arm}/{offset}'
      assert move==old['parameter_moves']['all'],'actual update mismatch'
      if arm=='CE_STRUCT':assert aux.bit_generator.state==old['aux_random_state']
      updates.append(dict(condition=condition,arm=arm,offset=offset,update=u,t=inp['t'],input_sha256=ih,actual_move=move,historical_exact=True,**rec));save('updates.json',updates)
      state=b.checkpoint(binding);assert state['updates']==state['ema']['updates']==u
      fr.atomic_checkpoint(O/f'{condition}_{arm}_offset{offset}.pt',dict(diagnostic_only=True,state=state,auxiliary_rng=aux.bit_generator.state if arm=='CE_STRUCT' else None,source_first=str(cp),protocol=P))
      evaluate(arm,offset);del before,delta,state
     state=b.checkpoint(binding);oldcp=Path(meta[condition]['checkpoint']) if arm=='CE' else S/f'{condition}_CE_STRUCT_temporary.pt';target=torch.load(oldcp,map_location='cpu')['state'];same={k:g.digest(state[k])==g.digest(target[k]) for k in ['model','optimizer','ema','rng','updates']};assert all(same.values()),'final full state mismatch';ends[arm]=g.digest(state['rng']);checks.append(dict(condition=condition,arm=arm,first=str(cp),last=str(oldcp),complete_endpoint_exact=same));save('checks.json',checks)
     status['completed'].append(condition+'_'+arm);save('progress.json',status);print(json.dumps(dict(completed=status['completed'][-1],updates=status['updates'])),flush=True);del parent,env,restored,state,target
    assert ends['CE']==ends['CE_STRUCT']
  del m,o,e,b;torch.cuda.empty_cache()
 fr.authenticate_sources()
 for path,h in w.read(O/'asset_binding.json').items():assert w.sha(path)==h
 assert (status['forwards'],status['backwards'],status['updates'])==(112,56,56);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(seconds=time.time()-started,end_epoch=time.time(),peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
