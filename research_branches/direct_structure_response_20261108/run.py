"""Eight DIRECT updates per parent, with immutable matched historical CE controls."""
import os,sys,json,time,traceback,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ['OBJECTIVE_DIAG_OUT'])
spec=importlib.util.spec_from_file_location('helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
sys.path.insert(0,str(ROOT/'research_branches/hard_structure_rebar_cpu_20261108'))
from objective import JointLocalScore
sys.path.insert(0,str(ROOT/'research_branches/hard_structure_incremental_cpu_20261108'))
sys.path.insert(0,str(ROOT/'research_branches/direct_structure_cpu_20261108'))
from score import DirectScore
OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2';REF=ROOT/'results/update_response_20261108T142622'
P=w.read(OUT/'protocol.json');start=time.time();stop=False;status=dict(status='running',updates=0,forwards=0,backwards=0,completed=[]);probes=w.read(OUT/'probes.json');updates=[];bindings=[]
def halt(*a):
 global stop
 stop=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
 if stop or time.time()>=P['science_stop_epoch']-15:raise TimeoutError('bounded stop')
def save(n,x):fr.save(OUT/n,x)
def main():
 fr.setup();fr.authenticate_sources();assert torch.cuda.is_available() and torch.cuda.device_count()==1
 # Existing runtime lambda is one; our only experimental coefficient is explicit below.
 assert w.update.__globals__['LAMBDA']==1.
 for path,sha in w.read(OUT/'source_binding.json').items():assert w.sha(path)==sha
 assets=w.read(OLD/'audit.json')['local_artifacts'];source_ref=w.read(ROOT/'results/hard_structure_gradient_gpu_20261108T180629/source_binding.json')
 for path,sha in source_ref.items():assert w.sha(path)==sha
 save('environment_before.json',g.common.environment('cuda'))
 for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
  ds,cfg=task['dataset'],task['row']['config'];check();data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  fixed=[]
  for t in P['times']:
   seed=w.seed32(P['evaluation_seeds'][ds],purpose='hard_structure_response_evaluation',time=t);inp=w.pair_inputs(case,cfg,10000+t,np.random.default_rng(seed),t)[0]
   fixed.append((t,seed,inp))
  torch.save(dict(inputs=fixed,purpose='reused diagnostic inputs independent of training stream, previously viewed development panel, not cross-graph validation'),OUT/f'{ds}_fixed_inputs.pt')
  m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);rng=np.random.default_rng(1)
  b=None
  def restore():
   nonlocal b
   g.restore_payload(w.cpu_copy(parent),m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();o.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,o,e,rng,'cuda',parent['updates']);live=b.checkpoint(binding)
   for k in ['model','optimizer','ema','rng','updates']:assert g.digest(live[k])==g.digest(parent[k]),'restore '+k
  def evaluate(arm):
   state=g.digest(b.checkpoint(binding))
   with b.validation():
    m.eval()
    for t,seed,inp in fixed:
     check();token=('hard_response',condition,arm,t);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
     try:
      with torch.no_grad():
       graph,logits,p=ctx.predict(inp['query'],inp['f'],t/1000,token,details=True);status['forwards']+=1;g.common.check_prob_device(p,inp['f'],case,m.prior)
       ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy();prob=p.cpu().double().numpy()
      assert np.isfinite(prob).all() and np.isfinite(ce).all();y=inp['labels'];core=JointLocalScore(case['space'].types,inp['query'],inp['f'],y,case['n_families'])
      with torch.no_grad():
       expected=core.features(torch.from_numpy(prob)).numpy();target=core.features(core.reference).numpy()
       evrng=np.random.default_rng(P['evaluation_mc_seeds'][ds]+t);samples=[]
       for offset in range(0,P['evaluation_draws'],8):
        bits=(evrng.random((8,len(prob)))<prob).astype(np.float64);samples.extend(core(torch.from_numpy(bits)).tolist())
      r=case['n_families'];preddeg=expected[:,:r].sum(1);truedeg=target[:,:r].sum(1);relations=[];nodegroups=[]
      for f in range(r):
       ix=inp['f']==f;pos=ix&(y==1);neg=ix&(y==0);nodes=np.isin(case['space'].types,case['space'].pairs[f])
       relations.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,expected_edges=float(prob[ix].sum()),true_edges=int(pos.sum()),missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum()),expected_relation_degree_MAE=float(abs(expected[nodes,f]-target[nodes,f]).mean()) if nodes.any() else None))
      for ty in np.unique(case['space'].types):
       ix=case['space'].types==ty;deg=truedeg[ix];low=float(np.quantile(deg,.25));high=float(np.quantile(deg,.9))
       # Frozen before looking at predictions: reference degree <=Q25 and >=Q90, disjoint middle; singleton groups retained.
       for name,mask in [('all',ix),('low',ix&(truedeg<=low)),('hub',ix&(truedeg>low)&(truedeg>=high)),('middle',ix&(truedeg>low)&(truedeg<high))]:
        nodegroups.append(dict(type=int(ty),group=name,support=int(mask.sum()),degree_MAE=float(abs(preddeg[mask]-truedeg[mask]).mean()) if mask.any() else None,triangle_participation_MAE=float(abs(expected[mask,r:]-target[mask,r:]).mean()) if mask.any() else None))
      row=dict(condition=condition,seed=task['seed'],arm=arm,updates=b.completed,t=t,readout='raw',input_sha256=w.input_hash([inp]),noise_seed=seed,CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[y==1].mean()),negative_CE=float(ce[y==0].mean()),expected_edges=float(prob.sum()),edge_abs_error=float(abs(prob.sum()-y.sum())),expected_degree_MAE=float(abs(preddeg-truedeg).mean()),expected_triangles=float(expected[:,r:].sum()/3),reference_triangles=float(target[:,r:].sum()/3),triangle_abs_error=float(abs(expected[:,r:].sum()-target[:,r:].sum())/3),expected_triangle_participation_MAE=float(abs(expected[:,r:]-target[:,r:]).mean()),hard_score_mean=float(np.mean(samples)),hard_score_samples=samples,relations=relations,nodegroups=nodegroups)
      probes.append(row);np.savez_compressed(OUT/f'{condition}_{arm}_t{t}.npz',probability=prob,CE=ce,expected_features=expected,reference_features=target);save('probes.json',probes)
     finally:ctx.close()
   assert g.digest(b.checkpoint(binding))==state,'evaluation polluted state/RNG'
  class Objective:
   def __init__(self,kind,aux):self.kind=kind;self.aux=aux;self.last={}
   def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
   def __call__(self,p,q,f,inp,*,logits):
    if self.kind=='CE':self.last=dict(objective_kind='CE');return p.sum()*0,p.new_empty(0)
    before_rng=g.digest(g.rng_capture(rng,'cuda'));prob=p.detach().cpu().double().numpy();bits=(self.aux.random(len(prob))<prob).astype(np.float64)
    direct=DirectScore(case['space'].types,q,f,inp['labels'],bits,case['n_families']);gradient=direct.logit_estimate(prob)
    assert np.isfinite(gradient).all();margin=logits[:,1]-logits[:,0];linear=(margin*torch.as_tensor(gradient,device=margin.device,dtype=margin.dtype)).sum()
    hard=float(direct.base);surrogate=P['coefficient']*(linear-linear.detach()+hard)
    assert g.digest(g.rng_capture(rng,'cuda'))==before_rng
    self.last=dict(objective_kind='direct_structure_conditional',aux_coefficient=P['coefficient'],hard_score_sample=hard,margin_gradient_L2=float(np.linalg.norm(gradient)),aux_random_state=self.aux.bit_generator.state)
    return surrogate,p.new_tensor([hard])
  with execution_policy(True):
   save('environment_active.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for parentarm in ['CE','PS']:
    condition=ds+'_'+parentarm;cp=OLD/f'{condition}_diagnostic_state.pt';sha=w.sha(cp);assert sha==assets[str(cp.relative_to(ROOT))];envelope=torch.load(cp,map_location='cpu');assert envelope['diagnostic_only'] and envelope['execution_policy']==VERSION and envelope['local_objective']==parentarm
    parent=envelope['state'];binding=parent['binding'];assert binding['config']==cfg and parent['updates']==task['start']+32
    restore()
    for t,seed,inp in fixed:
     oldprobe=next(x for x in probes if x['condition']==condition and x['arm']=='start' and x['t']==t);assert w.input_hash([inp])==oldprobe['input_sha256'],'fixed input mismatch'
    bindings.append(dict(condition=condition,seed=task['seed'],parent=str(cp),sha256=sha,data_sha256=w.sha(data),fixed_inputs=[dict(t=t,seed=seed,sha256=w.input_hash([inp])) for t,seed,inp in fixed]));save('assets.json',bindings)
    control_meta=w.read(OUT/'control_binding.json')[condition];assert control_meta['parent_sha256']==sha
    assert w.sha(control_meta['checkpoint'])==control_meta['checkpoint_sha256'];control=torch.load(control_meta['checkpoint'],map_location='cpu')
    assert control['local_objective']=='CE' and control['parent_sha256']==sha and control['execution_policy']==VERSION
    assert control['state']['binding']==binding and control['state']['updates']==parent['updates']+P['steps']
    arm_rngs={'CE':g.digest(control['state']['rng'])};del control
    for arm in ['CE_STRUCT']:
     restore();aux=np.random.default_rng(P['aux_seeds'][ds]);obj=Objective(arm,aux)
     for offset in range(1,P['steps']+1):
      check();u=b.completed+1;before={n:p.detach().cpu().clone() for n,p in m.named_parameters()};tick=time.perf_counter()
      with b.step(u):
       inp=g.one_input(case,cfg,u,rng);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256'];assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_before']
       status['forwards']+=1;rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=obj);status['updates']+=1;status['backwards']+=1
       assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_after']
      moves={}
      for group in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']:
       delta=torch.cat([(p.detach().cpu()-before[n]).flatten() for n,p in m.named_parameters() if group=='all' or w.category(n)==group]);moves[group]=w.stats(delta)
      updates.append(dict(condition=condition,arm=arm,offset=offset,update=u,t=inp['t'],input_sha256=ih,seconds=time.perf_counter()-tick,parameter_moves=moves,**rec));save('updates.json',updates)
      del before,delta
     evaluate(arm);state=b.checkpoint(binding);assert state['updates']==state['ema']['updates']==parent['updates']+P['steps'];arm_rngs[arm]=g.digest(state['rng'])
     fr.atomic_checkpoint(OUT/f'{condition}_{arm}_temporary.pt',dict(diagnostic_only=True,local_objective=arm,parent_sha256=sha,execution_policy=VERSION,protocol=P,auxiliary_rng=aux.bit_generator.state,state=state));status['completed'].append(condition+'_'+arm);save('progress.json',status)
     print(json.dumps(dict(completed=status['completed'][-1],updates=status['updates'],forwards=status['forwards'])),flush=True);del state
    assert arm_rngs['CE']==arm_rngs['CE_STRUCT'];restore();assert w.sha(cp)==sha
    del parent,envelope
  del m,o,e,b;torch.cuda.empty_cache()
 fr.authenticate_sources();assert (status['updates'],status['forwards'],status['backwards'])==(32,44,32);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(end_epoch=time.time(),seconds=time.time()-start,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
