"""Actual production AdamW one-step paired diagnostic, not formal training."""
import os,sys,json,time,traceback,signal,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(os.environ['OBJECTIVE_DIAG_OUT']);S=R/'results/direct_structure_response_20261108T205816'
spec=importlib.util.spec_from_file_location('helper',R/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(R/'research_branches/reproducible_aggregation_20261108'));from adapter import execution_policy,VERSION
sys.path.insert(0,str(R/'research_branches/direct_structure_cpu_20261108'));from score import DirectScore
P=w.read(O/'protocol.json');start=time.time();stop=False;rows=[];probes=[];status=dict(status='running',forwards=0,backwards=0,updates=0)
def signal_stop(*a):
 global stop
 stop=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,signal_stop)
def check():
 if stop or time.time()>=P['science_stop_epoch']-15:raise TimeoutError('ticket cutoff')
def save(n,x):fr.save(O/n,x)
def flat(xs,params):return torch.cat([(torch.zeros_like(p) if x is None else x).detach().cpu().flatten() for x,p in zip(xs,params)])
def geometry(a,b):return dict(dot=float(a.double()@b.double()),cosine=w.cosine(a,b))
def main():
 fr.setup();fr.authenticate_sources();assert torch.cuda.is_available() and torch.cuda.device_count()==1 and w.update.__globals__['LAMBDA']==1.
 for path,h in w.read(O/'source_binding.json').items():assert w.sha(path)==h
 binding_rows=w.read(S/'control_binding.json');old_updates=w.read(S/'updates.json')+w.read(S/'control_updates.json')
 for task in w.read(R/'results/output_boundary_replay_20261108T081654/tasks.json'):
  ds,cfg=task['dataset'],task['row']['config'];check();data=R/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  fixed=next(x[2] for x in torch.load(S/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs'] if x[0]==500)
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);rng=np.random.default_rng(1);named=list(m.named_parameters());params=[p for _,p in named]
  with execution_policy(True):
   save('environment.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for parentarm in ['CE','PS']:
    condition=ds+'_'+parentarm;meta=binding_rows[condition];assert w.sha(meta['parent'])==meta['parent_sha256'];envelope=torch.load(meta['parent'],map_location='cpu');parent=envelope['state'];binding=parent['binding'];assert binding['config']==cfg and envelope['execution_policy']==VERSION
    u=parent['updates']+1;b=None
    def restore():
     nonlocal b
     g.restore_payload(w.cpu_copy(parent),m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();o.zero_grad(set_to_none=True);b=g.TrainingBoundary(m,o,e,rng,'cuda',parent['updates']);live=b.checkpoint(binding)
     for k in ['model','optimizer','ema','rng','updates']:assert g.digest(live[k])==g.digest(parent[k]),'restore '+k
    def record_prob(arm,which,inp,prob,ce):
     y=inp['labels'];q=inp['query'];N=len(case['space'].types);A=np.zeros((N,N));A[q[0],q[1]]=A[q[1],q[0]]=prob;Y=np.zeros_like(A);Y[q[0],q[1]]=Y[q[1],q[0]]=y
     rel=[]
     for f in range(case['n_families']):
      pos=(inp['f']==f)&(y==1);neg=(inp['f']==f)&(y==0)
      rel.append(dict(family=f,positive_count=int(pos.sum()),negative_count=int(neg.sum()),positive_CE=float(ce[pos].mean()) if pos.any() else None,negative_CE=float(ce[neg].mean()) if neg.any() else None,missed_positive_mass=float((1-prob[pos]).sum()),false_positive_mass=float(prob[neg].sum())))
     row=dict(condition=condition,arm=arm,input=which,t=inp['t'],input_sha256=w.input_hash([inp]),CE=float(ce.sum()/case['space'].total),positive_CE=float(ce[y==1].mean()),negative_CE=float(ce[y==0].mean()),expected_edges=float(prob.sum()),expected_triangles=float((A*(A@A)).sum()/6),reference_triangles=float((Y*(Y@Y)).sum()/6),expected_degree_MAE=float(abs(A.sum(1)-Y.sum(1)).mean()),relations=rel)
     probes.append(row);np.savez_compressed(O/f'{condition}_{arm}_{which}.npz',probability=prob,CE=ce,query=q,families=inp['f'],labels=y,types=case['space'].types);save('probes.json',probes)
    def evaluate(arm,which,inp):
     check();before=g.digest(b.checkpoint(binding))
     with b.validation():
      m.eval();token=('one_step_eval',condition,arm,which);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
      try:
       with torch.no_grad():
        graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);status['forwards']+=1;g.common.check_prob_device(p,inp['f'],case,m.prior);ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy();prob=p.cpu().double().numpy()
       record_prob(arm,which,inp,prob,ce)
      finally:ctx.close()
     assert before==g.digest(b.checkpoint(binding)),'probe changed state'
    class Objective:
     def __init__(self,arm):self.arm=arm;self.last={};self.separated={}
     def weights(self,q,f,inp,*,device,dtype):return torch.ones(len(f),device=device,dtype=dtype)
     def __call__(self,p,q,f,inp,*,logits):
      prob=p.detach().cpu().double().numpy();ce_terms=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(f,device='cuda')]);record_prob(self.arm+'_before','training',inp,prob,ce_terms.detach().cpu().numpy())
      if self.arm=='CE':return p.sum()*0,p.new_empty(0)
      auxrng=np.random.default_rng(P['aux_seeds'][ds]);bits=(auxrng.random(len(prob))<prob).astype(float);direct=DirectScore(case['space'].types,q,f,inp['labels'],bits,case['n_families']);gradient=direct.logit_estimate(prob);margin=logits[:,1]-logits[:,0];linear=(margin*torch.as_tensor(gradient,device='cuda',dtype=margin.dtype)).sum();aux=P['coefficient']*(linear-linear.detach()+direct.base)
      for key,loss in [('CE',ce_terms.sum()/case['space'].total),('weighted_structure',aux)]:
       vals=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);status['backwards']+=1;self.separated[key]=flat(vals,params)
      self.last=dict(aux_coefficient=P['coefficient'],aux_seed=P['aux_seeds'][ds],hard_score_sample=direct.base)
      return aux,p.new_empty(0)
    deltas={};rngs={};gradients={};recs={}
    restore();base=torch.cat([p.detach().cpu().flatten() for p in params])
    # Reuse start t500 prediction with already authenticated identity; never relabel new output.
    oldrow=next(x for x in w.read(S/'probes.json') if x['condition']==condition and x['arm']=='start' and x['t']==500);assert w.input_hash([fixed])==oldrow['input_sha256']
    with np.load(S/f'{condition}_start_t500.npz') as z:record_prob('start','t500',fixed,z['probability'],z['CE'])
    for arm in ['CE','CE_STRUCT']:
     check();restore();obj=Objective(arm);captured={}
     def hook(*_):captured['gradient']=flat([p.grad for p in params],params)
     handle=o.register_step_pre_hook(hook)
     try:
      with b.step(u):
       inp=g.one_input(case,cfg,u,rng);assert w.input_hash([inp])==history[u-1]['input_sha256'];assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_before']
       rec=w.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b,objective=obj);status['updates']+=1;status['forwards']+=1;status['backwards']+=1
       assert g.digest(g.rng_capture(rng,'cuda'))==history[u-1]['full_rng_after']
     finally:handle.remove()
     delta=torch.cat([p.detach().cpu().flatten() for p in params])-base;deltas[arm]=delta;gradients[arm]=captured['gradient'];recs[arm]=rec
     old=next(x for x in old_updates if x['condition']==condition and x['arm']==arm and x['offset']==1)
     assert rec['unweighted_CE']==old['unweighted_CE'] and rec['gradient_norm']==old['gradient_norm'],'historical first-step scalar mismatch'
     assert w.stats(delta)==old['parameter_moves']['all'],'historical first-step actual movement mismatch'
     if arm=='CE_STRUCT':separated=obj.separated
     state=b.checkpoint(binding);assert state['updates']==state['ema']['updates']==u;rngs[arm]=g.digest(state['rng']);fr.atomic_checkpoint(O/f'{condition}_{arm}_temporary.pt',dict(diagnostic_only=True,parent_sha256=meta['parent_sha256'],state=state,aux_seed=P['aux_seeds'][ds] if arm=='CE_STRUCT' else None))
     evaluate(arm,'training',inp);evaluate(arm,'t500',fixed)
    assert rngs['CE']==rngs['CE_STRUCT']
    residual=gradients['CE_STRUCT']-separated['CE']-separated['weighted_structure'];difference=deltas['CE_STRUCT']-deltas['CE'];groups=[];lo=0;spans=[]
    for n,p in named:spans.append((n,w.category(n),lo,lo+p.numel()));lo+=p.numel()
    for group in ['all','shared_backbone_other','CN_exclusive','degree_branch','joint_head']:
     ix=[(a,z) for _,cat,a,z in spans if group=='all' or cat==group]
     cat=lambda v:torch.cat([v[a:z] for a,z in ix]);gs=cat(separated['weighted_structure']);gc=cat(separated['CE']);dc=cat(deltas['CE']);dt=cat(deltas['CE_STRUCT']);dd=dt-dc
     groups.append(dict(group=group,CE_move=w.stats(dc),DIRECT_move=w.stats(dt),incremental_move=w.stats(dd),move_cosine=w.cosine(dc,dt),weighted_structure_gradient=w.stats(gs),CE_gradient=w.stats(gc),structure_dot_CE_move=geometry(gs,dc),structure_dot_DIRECT_move=geometry(gs,dt),structure_dot_incremental_move=geometry(gs,dd),CE_dot_incremental_move=geometry(gc,dd)))
    torch.save(dict(parameter_names=[n for n,p in named],spans=spans,deltas=deltas,gradients=gradients,separated=separated),O/f'{condition}_tensors.pt')
    rows.append(dict(condition=condition,seed=task['seed'],training_t=inp['t'],parent_updates=parent['updates'],groups=groups,gradient_sum_residual=w.stats(residual),actual_first_step_matches_history=True,input_and_rng_match=True,updates=recs));save('results.json',rows);restore();assert w.sha(meta['parent'])==meta['parent_sha256'];print(json.dumps(dict(completed=condition,status=status)),flush=True)
    del parent,envelope,state,base,deltas,gradients,separated,residual,difference
  del m,o,e,b;torch.cuda.empty_cache()
 fr.authenticate_sources();assert (status['forwards'],status['backwards'],status['updates'])==(24,16,8);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(seconds=time.time()-start,end_epoch=time.time(),peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,completed=len(rows));save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
