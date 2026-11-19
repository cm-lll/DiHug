"""Four isolated CE+rank updates; reuse verified same-state CE/PS responses."""
import os,sys,json,time,signal,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ['OBJECTIVE_DIAG_OUT'])
OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2';REF=ROOT/'results/update_response_20261108T142622'
spec=importlib.util.spec_from_file_location('old_helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
spec2=importlib.util.spec_from_file_location('allocation_cpu_candidate',ROOT/'research_branches/local_allocation_design_20261108/objective.py');candidate=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(candidate)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
P=w.read(OUT/'protocol.json');start=time.time();stopped=False
status=dict(status='loading',start_epoch=start,updates=0,forwards=0,extra_gradients=0,completed=[])
def halt(*a):
 global stopped
 stopped=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
 if stopped or time.time()>=min(P['science_stop_epoch'],start+525):raise TimeoutError('budget boundary')
def save(n,x):fr.save(OUT/n,x)
def flat(v):return torch.cat([a.detach().cpu().flatten() for a in v])
def norm(v):return float(v.double().norm())
def dot(a,b):return float(a.double()@b.double())
def cos(a,b):return dot(a,b)/(norm(a)*norm(b)) if norm(a)*norm(b) else None
def main():
 fr.setup();fr.authenticate_sources()
 assert torch.cuda.is_available() and torch.cuda.device_count()==1 and '2080 Ti' in torch.cuda.get_device_name(0)
 env=dict(**g.common.environment('cuda'),execution_policy=VERSION);oldenv=w.read(REF/'environment.json')
 for k in ['python_version','torch_runtime_version','packages','cuda','device_name','execution_policy']:assert env[k]==oldenv[k],'environment mismatch '+k
 save('environment_before_strict.json',env)
 for path,sha in w.read(OUT/'source_binding.json').items():assert w.sha(path)==sha,'diagnostic source changed'
 artifacts=w.read(OLD/'audit.json')['local_artifacts'];oldresults=w.read(REF/'results.json');bindings=[];results=[]
 for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
  ds,cfg=task['dataset'],task['row']['config'];check()
  data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  fixed=torch.load(OLD/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs'];fixed={t:x for panel,t,seed,x in fixed if panel=='new_diagnostic'}
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);rng=np.random.default_rng(1)
  named=list(m.named_parameters());params=[p for n,p in named]
  def grads(loss):
   values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);status['extra_gradients']+=1
   return flat([torch.zeros_like(p) if x is None else x for p,x in zip(params,values)])
  def evaluate(inp):
   check();state=g.digest(b.checkpoint(binding));token=('rank_local',condition,inp['t'])
   with b.validation():
    m.eval();ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
     with torch.no_grad():
      graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);status['forwards']+=1
      g.common.check_prob_device(p,inp['f'],case,m.prior)
      ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy();prob=p.cpu().numpy()
     y=inp['labels'];rels=[];assert np.isfinite(ce).all() and np.isfinite(prob).all()
     for f in range(case['n_families']):
      ix=inp['f']==f;pos=ix&(y==1);neg=ix&(y==0);pred=np.zeros(len(case['space'].types));target=np.zeros_like(pred)
      for ends in inp['query']:np.add.at(pred,ends[ix],prob[ix]);np.add.at(target,ends[ix],y[ix])
      nodes=np.isin(case['space'].types,case['space'].pairs[f])
      rels.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum()),missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),expected_relation_degree_MAE=float(abs(pred[nodes]-target[nodes]).mean()) if nodes.any() else None))
     row=dict(t=inp['t'],input_sha256=w.input_hash([inp]),CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[y==1].mean()),negative_CE=float(ce[y==0].mean()),expected_edges=float(prob.sum()),relations=rels)
     np.savez_compressed(OUT/f'{condition}_RANK_fixed_{inp["t"]}.npz',probability=prob,CE=ce)
    finally:ctx.close()
   assert g.digest(b.checkpoint(binding))==state,'evaluation state contamination'
   return row
  with execution_policy(True):
   save('environment_active.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for parent_arm in ['CE','PS']:
    check();condition=ds+'_'+parent_arm;cp=OLD/f'{condition}_diagnostic_state.pt';assert w.sha(cp)==artifacts[str(cp.relative_to(ROOT))]
    envelope=torch.load(cp,map_location='cpu');assert envelope['local_objective']==parent_arm and envelope['execution_policy']==VERSION and envelope['diagnostic_only']
    parent=envelope['state'];binding=parent['binding'];assert binding['config']==cfg and parent['updates']==task['start']+32
    work=w.cpu_copy(parent);g.restore_payload(work,m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();o.zero_grad(set_to_none=True)
    b=g.TrainingBoundary(m,o,e,rng,'cuda',parent['updates']);live=b.checkpoint(binding)
    for k in ['model','optimizer','ema','rng','updates']:assert g.digest(live[k])==g.digest(parent[k]),'restore '+k
    previous=next(x for x in oldresults if x['condition']==condition)
    oldtensor=torch.load(REF/f'{condition}_local_tensors.pt',map_location='cpu');assert oldtensor['parameter_names']==[n for n,p in named]
    oldCEfile=REF/f'{condition}_CE_temporary.pt';oldCEsha=next(x['sha256'] for x in w.read(REF/'audit.json')['temporary_states'] if (ROOT/Path(x['path'])).resolve()==oldCEfile);assert w.sha(oldCEfile)==oldCEsha
    oldCEstate=torch.load(oldCEfile,map_location='cpu')['state'];u=b.completed+1;before=flat(params);captured={};collected={}
    class Objective:
     last={}
     def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
     def __call__(self,p,q,f,inp,*,logits):
      assert logits.ndim==2 and logits.shape[1]==2
      rank=candidate.CEUnitsRank(case['space'].types,case['space'].pairs,case['prior'],q,f,inp['labels'],schema_slots=case['space'].total)
      aux,detail=rank(logits[:,1]-logits[:,0]);assert torch.isfinite(aux)
      ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(f,device='cuda')]).sum()/case['space'].total
      state_before=g.digest(o.state_dict());rng_before=g.digest(g.rng_capture(rng,'cuda'))
      collected['CE']=grads(ce);collected['rank']=grads(aux)
      assert g.digest(o.state_dict())==state_before and g.digest(g.rng_capture(rng,'cuda'))==rng_before
      assert all(p.grad is None for p in params),'autograd polluted training gradients'
      collected['gradient_none_before_main']=[p.grad is None for p in params]
      self.last=dict(objective_kind='CEUnitsRank',coefficient=1.,eligible_groups=len(rank.groups),excluded_groups=len(rank.excluded))
      return aux,torch.stack([v.mean() for v in detail]) if detail else p.new_empty(0)
    def capture(*args):
     captured['gradient']=flat([torch.zeros_like(p) if p.grad is None else p.grad for p in params]);captured['gradient_none']=[p.grad is None for p in params]
    hook=o.register_step_pre_hook(capture);tick=time.perf_counter()
    try:
     with b.step(u):
      inp=g.one_input(case,cfg,u,rng);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256']==previous['train_before']['input_sha256']
      assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_before']
      rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=Objective());status['updates']+=1;status['forwards']+=1
      assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_after']
    finally:hook.remove()
    torch.cuda.synchronize();seconds=time.perf_counter()-tick;delta=flat(params)-before
    assert torch.allclose(collected['CE'],oldtensor['actual_gradients']['CE'],atol=2e-6,rtol=1e-5),'reused CE gradient mismatch'
    assert torch.allclose(collected['CE']+collected['rank'],captured['gradient'],atol=2e-6,rtol=1e-5),'gradient reconstruction mismatch'
    assert abs(rec['unweighted_CE']-previous['outcomes']['CE']['training_record']['unweighted_CE'])<1e-7
    state=b.checkpoint(binding);assert g.digest(state['rng'])==g.digest(oldCEstate['rng']) and state['updates']==state['ema']['updates']==u
    assert captured['gradient_none']==previous['outcomes']['CE']['gradient_none']
    for k in state['optimizer']['state']:assert torch.equal(state['optimizer']['state'][k]['step'],oldCEstate['optimizer']['state'][k]['step'])
    responses={t:evaluate(fixed[t]) for t in [1,500,1000]}
    geometry={};dd=delta-oldtensor['displacements']['CE'];spans=oldtensor['spans']
    for group in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']:
     ix=[(lo,hi) for n,c,lo,hi in spans if group=='all' or c==group];cat=lambda v:torch.cat([v[lo:hi] for lo,hi in ix])
     geometry[group]=dict(CE_gradient_L2=norm(cat(collected['CE'])),rank_gradient_L2=norm(cat(collected['rank'])),gradient_cosine=cos(cat(collected['CE']),cat(collected['rank'])),CE_displacement_L2=norm(cat(oldtensor['displacements']['CE'])),rank_displacement_L2=norm(cat(delta)),displacement_cosine=cos(cat(delta),cat(oldtensor['displacements']['CE'])),difference_L2=norm(cat(dd)),current_CE_linear_rank=dot(cat(collected['CE']),cat(delta)),current_CE_linear_CE=dot(cat(collected['CE']),cat(oldtensor['displacements']['CE'])))
    linear={}
    for label,metric in [('eval_positive','positive_CE'),('eval_negative','negative_CE')]:
     pred=dot(oldtensor['evaluation_gradients'][label],dd);actual=responses[500][metric]-previous['outcomes']['CE']['responses']['500'][metric]
     linear[label]=dict(rank_minus_CE_actual=actual,linear_prediction=pred,remainder=actual-pred)
    results.append(dict(condition=condition,dataset=ds,seed=task['seed'],parent_branch=parent_arm,update=u,training_t=inp['t'],training_record=rec,responses=responses,geometry=geometry,linear_t500=linear,seconds=seconds,CE_gradient_reference_max_abs=float((collected['CE']-oldtensor['actual_gradients']['CE']).abs().max()),gradient_reconstruction_max_abs=float((captured['gradient']-collected['CE']-collected['rank']).abs().max()),current_train_after_not_evaluated=True))
    save('results.json',results)
    fr.atomic_checkpoint(OUT/f'{condition}_RANK_temporary.pt',dict(diagnostic_only=True,intervention='CEUnitsRank',parent_sha256=w.sha(cp),execution_policy=VERSION,state=state))
    torch.save(dict(spans=spans,gradients=collected,actual_gradient=captured['gradient'],displacement=delta,gradient_none=captured['gradient_none']),OUT/f'{condition}_tensors.pt')
    bindings.append(dict(condition=condition,parent=str(cp),sha256=w.sha(cp),reused_CE_state=str(oldCEfile),reused_CE_sha256=oldCEsha,input_sha256=ih,update=u,data_sha256=w.sha(data)));save('assets.json',bindings)
    assert w.sha(cp)==artifacts[str(cp.relative_to(ROOT))];status['completed'].append(condition);save('progress.json',status)
    del oldtensor,oldCEstate,parent,envelope,state,collected,captured,delta,live,work
  del m,o,e,b
  torch.cuda.empty_cache()
 fr.authenticate_sources();assert (status['updates'],status['forwards'],status['extra_gradients'])==(4,16,8);status['status']='complete'
if __name__=='__main__':
 try:main()
 except BaseException as exc:status.update(status='stopped_or_failed',error=repr(exc),traceback=traceback.format_exc())
 finally:
  status.update(end_epoch=time.time(),seconds=time.time()-start,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',status);print(json.dumps(status),flush=True)
 raise SystemExit(0 if status['status']=='complete' else 2)
