"""Frozen DIRECT extension: eight registered trajectories, no new objective."""
import os,sys,json,time,signal,traceback,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(os.environ['OBJECTIVE_DIAG_OUT'])
spec=importlib.util.spec_from_file_location('helper',R/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(R/'research_branches/reproducible_aggregation_20261108'));from adapter import execution_policy,VERSION
sys.path.insert(0,str(R/'research_branches/direct_structure_cpu_20261108'));from score import DirectScore
P=w.read(O/'protocol.json');began=time.time();stop=False
status=dict(status='running',updates=0,forwards=0,backwards=0,completed=[])
updates=[];probes=[];checks=[];b=None;binding=None;aux=None;condition=None;arm=None
def halt(*a):
 global stop
 stop=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def save(name,x):fr.save(O/name,x)
def check():
 lease=w.read(O/'lease.json')
 if stop or time.time()>=min(P['science_stop_epoch'],lease['hard_stop_epoch'])-25:raise TimeoutError('authorized deadline')
def main():
 global b,binding,aux,condition,arm
 fr.setup();fr.authenticate_sources()
 assert w.update.__globals__['LAMBDA']==1.
 for p,h in w.read(O/'source_binding.json').items():assert w.sha(p)==h,p
 for p,h in w.read(O/'asset_binding.json').items():assert w.sha(p)==h,p
 tasks=w.read(R/'results/output_boundary_replay_20261108T081654/tasks.json')
 # Freeze all actual evaluation states before any model construction/prediction.
 prepared={};ib=[]
 for task in tasks:
  ds,cfg=task['dataset'],task['row']['config'];data=R/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  fixed=[]
  for j,item in enumerate(P['evaluation'][ds]['inputs']):
   inp=w.pair_inputs(case,cfg,10000+j,np.random.default_rng(item['noise_seed']),item['t'])[0]
   assert inp['t']==item['t'] and len(inp['labels'])==len(inp['f'])
   ih=w.input_hash([inp]);fixed.append(dict(**item,index=j,input=inp,input_sha256=ih));ib.append(dict(dataset=ds,index=j,**item,sha256=ih))
  old=next(x[2] for x in torch.load(R/'results/direct_structure_response_20261108T205816'/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs'] if x[0]==500)
  torch.save(dict(fixed=fixed,legacy=old,types=case['space'].types,relations=case['n_families'],data_sha256=w.sha(data)),O/f'{ds}_inputs.pt')
  prepared[ds]=(case,cfg,fixed,old)
 save('input_binding.json',ib)
 assert len(ib)==40 and torch.cuda.is_available() and torch.cuda.device_count()==1
 for task in tasks:
  ds=task['dataset'];case,cfg,fixed,legacy=prepared[ds]
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  m=g.StructuralDenoiser(case,cfg,'A').cuda();opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=g.ParameterEMA(m);rng=np.random.default_rng(1)
  def restore(payload):
   global b,binding
   binding=payload['binding'];assert binding['config']==cfg
   g.restore_payload(w.cpu_copy(payload),m,opt,ema,binding,lambda x:g.rng_restore(x,rng,'cuda'))
   m.train();opt.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,opt,ema,rng,'cuda',payload['updates'])
   live=b.checkpoint(binding);same={k:g.digest(live[k])==g.digest(payload[k]) for k in ['model','optimizer','ema','rng','updates']};assert all(same.values())
   checks.append(dict(condition=condition,arm=arm,restored_update=b.completed,state_exact=same));save('checks.json',checks)
  def evaluate(offset,which='new'):
   check();before=g.digest(b.checkpoint(binding));items=fixed if which=='new' else [dict(index=0,t=500,input=legacy,input_sha256=w.input_hash([legacy]))]
   with b.validation():
    m.eval()
    for item in items:
     check();inp=item['input'];t=item['t'];token=('medium',condition,arm,offset,which,item['index']);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
     try:
      tick=time.time()
      with torch.no_grad():
       graph,logits,p=ctx.predict(inp['query'],inp['f'],t/1000,token,details=True);status['forwards']+=1;g.common.check_prob_device(p,inp['f'],case,m.prior)
       ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy();prob=p.cpu().double().numpy()
      assert np.isfinite(prob).all() and np.isfinite(ce).all()
      name=f'{condition}_{arm}_o{offset}_{which}_{item["index"]}.npz';np.savez_compressed(O/name,probability=prob,CE=ce)
      probes.append(dict(condition=condition,dataset=ds,seed=task['seed'],arm=arm,offset=offset,update=b.completed,panel=which,index=item['index'],t=t,path=name,input_sha256=item['input_sha256'],seconds=time.time()-tick))
     finally:ctx.close()
   assert before==g.digest(b.checkpoint(binding)),'evaluation contaminated full state'
   save('probes.json',probes);save('progress.json',status)
  class Objective:
   def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
   def __call__(self,p,q,f,inp,*,logits):
    if arm=='CE':self.last={};return p.sum()*0,p.new_empty(0)
    prior=g.digest(g.rng_capture(rng,'cuda'));prob=p.detach().cpu().double().numpy();bits=(aux.random(len(prob))<prob).astype(float)
    score=DirectScore(case['space'].types,q,f,inp['labels'],bits,case['n_families']);grad=score.logit_estimate(prob);assert np.isfinite(grad).all()
    margin=logits[:,1]-logits[:,0];linear=(margin*torch.as_tensor(grad,device='cuda',dtype=margin.dtype)).sum()
    self.last=dict(hard_score_sample=score.base,aux_random_state=aux.bit_generator.state,margin_gradient_L2=float(np.linalg.norm(grad)))
    assert prior==g.digest(g.rng_capture(rng,'cuda'))
    return P['coefficient']*(linear-linear.detach()+score.base),p.new_tensor([score.base])
  with execution_policy(True):
   save('environment.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for parent_kind in ['CE','PS']:
    condition=ds+'_'+parent_kind;specs=[x for x in P['tasks'] if x['condition']==condition];assert len(specs)==2
    arm='start';origin=torch.load(specs[0]['common_origin'],map_location='cpu');assert origin['diagnostic_only'] and origin['execution_policy']==VERSION
    restore(origin['state']);assert b.completed==task['start']+32;evaluate(0);del origin
    ends={};contracts={}
    for s in specs:
     arm=s['arm'];check();env=torch.load(s['resume_path'],map_location='cpu');assert env['diagnostic_only'] and env['local_objective']==arm and env['execution_policy']==VERSION
     assert env['parent_sha256']==s['common_origin_sha256'];restore(env['state']);assert b.completed==task['start']+40
     aux=np.random.default_rng(1)
     if arm=='CE_STRUCT':
      aux.bit_generator.state=env['auxiliary_rng']
      original=next(x for x in w.read(R/'results/direct_structure_response_20261108T205816/updates.json') if x['condition']==condition and x['offset']==8)
      assert aux.bit_generator.state==original['aux_random_state']
     evaluate(8);obj=Objective();local=[]
     for offset in range(9,65):
      check();u=b.completed+1;tick=time.time()
      with b.step(u):
       inp=g.one_input(case,cfg,u,rng);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256']
       assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_before']
       rec=w.update(m,opt,case,inp,1000,'cuda',ema,u-1,transaction=b,objective=obj);status['updates']+=1;status['forwards']+=1;status['backwards']+=1
       assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_after']
      local.append(ih);updates.append(dict(condition=condition,arm=arm,offset=offset,update=u,t=inp['t'],input_sha256=ih,wall_seconds=time.time()-tick,**rec))
      save('updates.json',updates)
      if offset in [32,64]:
       state=b.checkpoint(binding);assert state['updates']==state['ema']['updates']==u
       fr.atomic_checkpoint(O/f'{condition}_{arm}_offset{offset}.pt',dict(diagnostic_only=True,state=state,auxiliary_rng=aux.bit_generator.state if arm=='CE_STRUCT' else None,source_resume=s['resume_path'],protocol=P,execution_policy=VERSION))
       evaluate(offset);evaluate(offset,'legacy');del state
      if offset%8==0:
       save('progress.json',dict(**status,current=condition+'_'+arm,offset=offset,elapsed=time.time()-began))
      # Cost check uses actual counted work, does not add trial updates.
      if status['updates']==8:
       median=float(np.median([x['wall_seconds'] for x in updates]));remaining_updates=448-status['updates'];remaining_forwards=1024-status['forwards']-remaining_updates
       est=remaining_updates*max(3.,median*1.25)+remaining_forwards*1.2+330
       remaining=min(P['science_stop_epoch'],w.read(O/'lease.json')['hard_stop_epoch'])-time.time()
       save('throughput.json',dict(median_update_seconds=median,remaining_estimate_seconds=est,remaining_budget_seconds=remaining))
       if est>remaining:raise TimeoutError('full panel cost exceeds remaining budget')
     ends[arm]=g.digest(b.checkpoint(binding)['rng']);contracts[arm]=local
     status['completed'].append(condition+'_'+arm);save('progress.json',status);print(json.dumps(dict(completed=status['completed'][-1],updates=status['updates'],forwards=status['forwards'])),flush=True);del env
    assert ends['CE']==ends['CE_STRUCT'] and contracts['CE']==contracts['CE_STRUCT'],'arm input/RNG pairing'
  del m,opt,ema;b=None;torch.cuda.empty_cache()
 fr.authenticate_sources()
 for p,h in w.read(O/'asset_binding.json').items():assert w.sha(p)==h
 assert (status['updates'],status['forwards'],status['backwards'])==(448,1024,448)
 assert len(probes)==576;status['status']='complete'
try:main()
except BaseException as exc:
 status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
 if b is not None:
  try:fr.atomic_checkpoint(O/'interrupted.pt',dict(diagnostic_only=True,state=b.checkpoint(binding),auxiliary_rng=aux.bit_generator.state if aux is not None else None,condition=condition,arm=arm,protocol=P));status['recoverable_saved']=True
  except BaseException as err:status['recoverable_saved']=False;status['save_error']=repr(err)
finally:
 save('probes.json',probes);save('updates.json',updates)
 status.update(end_epoch=time.time(),seconds=time.time()-began,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
