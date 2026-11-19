"""Bounded same-state AdamW response: eight independent proposals, no training run."""
import os,sys,json,time,signal,importlib.util,traceback,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_response_20261109'
PLAN=ROOT/'results/joint_degree_response_plan_20261109';OLD=ROOT/'results/hard_structure_response_20261108T182154'
PREV=ROOT/'results/joint_degree_parameter_signal_20261109';REL=ROOT/'results/joint_degree_relation_signal_20261109'
spec=importlib.util.spec_from_file_location('archived_helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
P=json.loads((OUT/'protocol.json').read_text());START=time.time();STOP=False;BIND={}
status=dict(status='preparing',forwards=0,backwards=0,optimizer_steps=0,new_graphs=0,completed=[])
def save(n,x):fr.save(OUT/(n+'.json'),x)
def bind(p):p=Path(p);BIND[str(p)]=w.sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def halt(*_):
 global STOP
 STOP=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
 if STOP or time.time()>P['science_stop_epoch']-15:raise TimeoutError('fixed science deadline')
def compare(a,b):
 a=a.double();b=b.double();den=float(a.norm()*b.norm())
 return dict(norm_a=float(a.norm()),norm_b=float(b.norm()),cosine=float(a@b)/den if den else None,difference_norm=float((a-b).norm()))
def main():
 check();fr.setup();fr.authenticate_sources();bind(__file__)
 for p,h in read(PLAN/'binding.json').items():assert w.sha(p)==h;BIND[p]=h
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 assets=read(PLAN/'assets.json');oldenv=read(OLD/'environment_active.json')
 fixed=torch.load(bind(OLD/'PubMed_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
 inp=next(v[2] for v in fixed if v[0]==1000);oldhashes={w.input_hash([v[2]]) for v in fixed}
 allstate=[];allchecks=[];allmoves=[];evalinputs=None
 with execution_policy(True):
  env=g.common.environment('cuda');save('environment',dict(**env,execution_policy=VERSION))
  for k in ['python_version','torch_runtime_version','device_name','packages','float32_matmul_precision','matmul_allow_tf32','deterministic_algorithms','deterministic_warn_only','torch_threads']:assert env[k]==oldenv[k],k
  for asset in assets:
   check();parent=asset['parent'];envelope=torch.load(bind(asset['checkpoint']),map_location='cpu',weights_only=False);state=envelope['state'];binding=state['binding'];cfg=binding['config']
   assert w.sha(asset['checkpoint'])==asset['sha256'] and state['updates']==state['ema']['updates']==2040 and envelope['execution_policy']==VERSION
   data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets']['PubMed']['file'];assert w.sha(bind(data))==cfg['datasets']['PubMed']['sha256']
   with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
   if evalinputs is None:
    evalinputs={};meta=[]
    for key,seed in asset['evaluation_noise_seeds'].items():
     t=int(key.split('_')[0][1:]);v=w.pair_inputs(case,cfg,10000+t,np.random.default_rng(seed),t)[0]
     h=w.input_hash([v]);duplicates=[x['key'] for x in meta if x['hash']==h]
     audited=next(x for x in read(OUT/'input_collision_audit.json') if x['key']==key)
     assert h==audited['hash'] and audited['rng_consumed'] and seed==audited['seed']
     evalinputs[key]=v;meta.append(dict(key=key,t=t,noise_seed=seed,hash=h,previously_observed=h in oldhashes,duplicate_of=duplicates[0] if duplicates else None))
    torch.save(dict(inputs=evalinputs,metadata=meta),OUT/'evaluation_inputs.pt');save('evaluation_inputs',meta)
    np.savez_compressed(OUT/'schema.npz',types=case['space'].types,query=inp['query'],relations=inp['f'],reference=inp['labels'])
   m=g.StructuralDenoiser(case,cfg,'A').cuda();opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});ema=g.ParameterEMA(m);rng=np.random.default_rng(1)
   def restore():
    g.restore_payload(w.cpu_copy(state),m,opt,ema,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();opt.zero_grad(set_to_none=True)
    b=g.TrainingBoundary(m,opt,ema,rng,'cuda',2040);live=b.checkpoint(binding)
    assert all(g.digest(live[k])==g.digest(state[k]) for k in ['model','optimizer','ema','rng','updates'])
    return b
   boundary=restore();names=[];params=[]
   for n,pa in m.named_parameters():
    if pa.requires_grad:names.append(n);params.append(pa)
   layout=read(PREV/f'{parent}_parameter_layout.json');assert names==[r['name'] for r in layout]
   def flatten():return torch.cat([pa.detach().cpu().reshape(-1) for pa in params])
   def writeflat(v):
    with torch.no_grad():
     for pa,r in zip(params,layout):pa.copy_(v[r['start']:r['end']].reshape(pa.shape).to(pa.device))
   def record(arm,key,v,logits,p):
    terms=w.supported_binary_ce(logits,torch.as_tensor(v['labels'],device='cuda'),m.prior[torch.as_tensor(v['f'],device='cuda')])
    assert torch.isfinite(terms).all()
    np.savez_compressed(OUT/f'{parent}_{arm}_{key}.npz',probability=p.detach().cpu().numpy(),ce=terms.detach().cpu().numpy(),query=v['query'],relation=v['f'],reference=v['labels'],t=v['t'])
   def forward(v,arm,key,grad=False):
    check();m.eval();token=(parent,arm,key);ctx=g.FrozenTypedContext(m,case['space'].types,v['edges'],v['edge_families'],token)
    try:
     with torch.set_grad_enabled(grad):graph,logits,p=ctx.predict(v['query'],v['f'],v['t']/1000,token,details=True)
     status['forwards']+=1;g.common.check_prob_device(p,v['f'],case,m.prior);record(arm,key,v,logits,p)
     return logits,p
    finally:ctx.close()
   base=flatten();before=boundary.checkpoint(binding)
   vectors=np.load(bind(REL/f'{parent}_logit_decomposition.npz'))
   with boundary.validation():
    logits,p=forward(inp,'NOOP','update',True)
    with np.load(ROOT/f'results/joint_degree_full_signal_20261109/PubMed_{parent}_CE_t1000_draw0.npz') as z:assert np.array_equal(p.detach().cpu().double().numpy(),z['probability'])
    ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).sum()/case['space'].total
    objectives=[('CE',ce)]+[(name,((logits[:,1]-logits[:,0])*torch.as_tensor(vectors[k],device='cuda',dtype=logits.dtype)).sum()) for name,k in [('FULL','total'),('COMMON','common'),('RESIDUAL','residual')]]
    grads={};nones={}
    for j,(name,objective) in enumerate(objectives):
     check();vs=torch.autograd.grad(objective,params,retain_graph=j<3,allow_unused=True);status['backwards']+=1
     assert all(v is None or torch.isfinite(v).all() for v in vs)
     nones[name]=[n for n,v in zip(names,vs) if v is None]
     grads[name]=torch.cat([(torch.zeros_like(pa) if v is None else v).detach().cpu().flatten() for pa,v in zip(params,vs)])
     assert all(pa.grad is None for pa in params)
   after=boundary.checkpoint(binding);assert all(g.digest(before[k])==g.digest(after[k]) for k in ['model','optimizer','ema','rng','updates'])
   previous=torch.load(bind(PREV/f'{parent}_parameter_gradients.pt'),map_location='cpu',weights_only=False)
   for key,prior in [('CE','CE'),('FULL','new_mean')]:assert torch.allclose(grads[key],previous['gradients'][prior],atol=cfg['tolerances']['fp32_atol'],rtol=cfg['tolerances']['fp32_rtol'])
   assert all(set(v)==set(asset['missing_optimizer_state_names']) for v in nones.values())
   allchecks.append(dict(parent=parent,p_exact=True,state_unchanged=True,None_preserved=nones,gradient_linearity=compare(grads['FULL'],grads['COMMON']+grads['RESIDUAL']),lambda_value=asset['proposed_local_unit_balance_lambda']))
   save('checks',allchecks)
   del logits,p,ce,objective,objectives,vs,previous
   positions={'NOOP':base};lam=asset['proposed_local_unit_balance_lambda']
   for arm in ['CE','FULL','COMMON','RESIDUAL']:
    check();boundary=restore();gv=grads['CE'] if arm=='CE' else grads['CE']+lam*grads[arm]
    for n,pa,r in zip(names,params,layout):pa.grad=None if n in nones['CE'] else gv[r['start']:r['end']].reshape(pa.shape).to(pa.device).clone()
    rng_before=g.digest(g.rng_capture(rng,'cuda'));buffers_before=g.digest({n:v.detach().cpu() for n,v in m.named_buffers()})
    ema.at_update_boundary=False;opt.step();status['optimizer_steps']+=1;ema.update(m,2041);ema.at_update_boundary=True
    assert all(torch.isfinite(pa).all() for pa in params) and ema.updates==2041
    assert g.digest(g.rng_capture(rng,'cuda'))==rng_before and buffers_before==g.digest({n:v.detach().cpu() for n,v in m.named_buffers()})
    od=w.cpu_copy(opt.state_dict());assert len(od['state'])==472 and all(float(x['step'])==2041 and torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all() for x in od['state'].values())
    positions[arm]=flatten();allstate.append(dict(parent=parent,arm=arm,optimizer_digest=g.digest(od),EMA_digest=g.digest(ema.state_dict()),rng_unchanged=True,buffers_unchanged=True,completed_updates=2041,diagnostic_only=True))
    save('state_records',allstate);del od
   uf=positions['FULL'].double()-positions['CE'];ur=positions['RESIDUAL'].double()-positions['CE'];nf=float(uf.norm());nr=float(ur.norm())
   larger='FULL' if nf>=nr else 'RESIDUAL';rho=min(nf,nr);den=max(nf,nr);ratio=rho/den if den else 0.
   positions['MATCH']=(positions['CE'].double()+ratio*(positions[larger].double()-positions['CE'])).float()
   save(parent+'_matching',dict(larger=larger,radius=rho,ratio=ratio,full_aux_norm=nf,residual_aux_norm=nr,actual_aux_norm=float((positions['MATCH'].double()-positions['CE']).norm()),synthetic_parameters_only=True,degenerate=den==0 or rho==0))
   for arm,v in positions.items():
    move=v.double()-base;aux=v.double()-positions['CE']
    allmoves.append(dict(parent=parent,arm=arm,**compare(move,positions['CE'].double()-base),aux_norm=float(aux.norm())))
   save('movements',allmoves)
   torch.save(dict(diagnostic_only=True,not_resumable=True,base_sha256=asset['sha256'],positions=positions,layout=layout),OUT/f'{parent}_positions.pt')
   restore();m.eval();probe_rng=g.digest(g.rng_capture(rng,'cuda'));buffer_digest=g.digest({n:v.detach().cpu() for n,v in m.named_buffers()});opt_digest=g.digest(opt.state_dict());ema_digest=g.digest(ema.state_dict())
   for arm,v in positions.items():
    writeflat(v)
    for key,ei in evalinputs.items():
     meta=next(x for x in read(OUT/'evaluation_inputs.json') if x['key']==key)
     if meta['duplicate_of']:
      import shutil
      shutil.copy2(OUT/f'{parent}_{arm}_{meta["duplicate_of"]}.npz',OUT/f'{parent}_{arm}_{key}.npz')
     else:forward(ei,arm,key)
    if arm!='NOOP':forward(inp,arm,'update')
    assert g.digest(g.rng_capture(rng,'cuda'))==probe_rng and g.digest(opt.state_dict())==opt_digest and g.digest(ema.state_dict())==ema_digest
    assert g.digest({n:x.detach().cpu() for n,x in m.named_buffers()})==buffer_digest
    save('progress',dict(**status,parent=parent,last_arm=arm,elapsed=time.time()-START))
   restore();status['completed'].append(parent);print(json.dumps(status),flush=True)
   del m,opt,ema,boundary,params,grads,positions,before,after,envelope,state,base
   gc.collect();torch.cuda.empty_cache()
 assert status['forwards']==72 and status['backwards']==8 and status['optimizer_steps']==8
 fr.authenticate_sources();assert all(w.sha(p)==h for p,h in BIND.items());save('binding',BIND);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(worker_seconds=time.time()-START,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0.)
 save('receipt',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
