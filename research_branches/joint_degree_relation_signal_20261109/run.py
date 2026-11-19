"""Relation-common VJP, residual by subtraction from archived total; no updates."""
import os,sys,json,time,signal,hashlib,importlib.util,traceback,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_degree_relation_signal_20261109'
PREV=ROOT/'results/joint_degree_parameter_signal_20261109'
OLD=ROOT/'results/hard_structure_response_20261108T182154'
FULL=ROOT/'results/joint_degree_full_signal_20261109'
spec=importlib.util.spec_from_file_location('archived_helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
P=json.loads((OUT/'protocol.json').read_text());START=time.time();BIND={};STOP=False
status=dict(status='preparing',forwards=0,gradient_queries=0,optimizer_updates=0,new_graphs=0,completed=[])
def save(n,x):fr.save(OUT/(n+'.json'),x)
def bind(p):
 p=Path(p);BIND[str(p)]=w.sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def halt(*args):
 global STOP
 STOP=True
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
def check():
 if STOP or time.time()>=P['science_stop_epoch']-10:raise TimeoutError('bounded diagnostic stop')
def compare(x,y):
 xx=x.double();yy=y.double();den=float(xx.norm()*yy.norm());avg=float((xx.norm()+yy.norm())/2)
 return dict(cosine=float(xx@yy/den) if den else None,norm_left=float(xx.norm()),norm_right=float(yy.norm()),
  relative_difference=float((xx-yy).norm()/avg) if avg else None,
  mean_norm_over_average_norm=float(((xx+yy)/2).norm()/avg) if avg else None,
  opposite_nonzero=int(((xx*yy)<0).sum()),both_nonzero=int(((xx!=0)&(yy!=0)).sum()))
def main():
 check();fr.setup();fr.authenticate_sources()
 for file in [Path(__file__),ROOT/'research_branches/reproducible_aggregation_20261108/adapter.py',ROOT/'results/output_boundary_replay_20261108T081654/worker.py']:bind(file)
 for name in ['source_binding.json']:
  for p,h in read(OLD/name).items():assert w.sha(p)==h;bind(p)
 for p,h in read(ROOT/'results/hard_structure_gradient_gpu_20261108T180629/source_binding.json').items():assert w.sha(p)==h;bind(p)
 assert torch.cuda.is_available() and torch.cuda.device_count()==1 and '2080 Ti' in torch.cuda.get_device_name(0)
 save('environment_before',g.common.environment('cuda'))
 oldenv=read(OLD/'environment_active.json');statechecks=[];matching=[];summary=[];byparam=[]
 fixed=torch.load(bind(OLD/'PubMed_fixed_inputs.pt'),map_location='cpu',weights_only=False)['inputs']
 t,noise_seed,inp=next(v for v in fixed if v[0]==1000)
 probes=read(OLD/'probes.json')
 source_verified=read(FULL/'verified_prior_bindings.json')
 assert source_verified[str(OLD/'PubMed_fixed_inputs.pt')]==BIND[str(OLD/'PubMed_fixed_inputs.pt')]
 with execution_policy(True):
  actual=g.common.environment('cuda');save('environment_active',dict(**actual,execution_policy=VERSION))
  for k in ['python_version','torch_runtime_version','device_name','packages','float32_matmul_precision','matmul_allow_tf32','deterministic_algorithms','deterministic_warn_only','torch_threads']:
   assert actual[k]==oldenv[k],('environment',k,actual[k],oldenv[k])
  for parent in ['CE','PS']:
   check();tag=f'PubMed_{parent}_CE_t1000';cp=OLD/f'PubMed_{parent}_CE_temporary.pt'
   expected=read(cp.with_suffix('.sha256.json'))['sha256'];bind(cp);assert BIND[str(cp)]==expected
   env=torch.load(cp,map_location='cpu',weights_only=False)
   assert env['diagnostic_only'] and env['local_objective']=='CE' and env['execution_policy']==VERSION
   state=env['state'];binding=state['binding'];cfg=binding['config']
   assert state['updates']==state['ema']['updates']==2040 and binding['dataset']=='PubMed' and binding['train_seed']==2026110422
   asset=next(r for r in read(OLD/'assets.json') if r['condition']==f'PubMed_{parent}')
   assert env['parent_sha256']==asset['sha256']
   pr=next(r for r in probes if r['condition']==f'PubMed_{parent}' and r['arm']=='CE' and r['t']==1000)
   assert pr['readout']=='raw' and pr['updates']==2040 and pr['input_sha256']==w.input_hash([inp]) and pr['noise_seed']==noise_seed
   data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets']['PubMed']['file']
   bind(data);assert BIND[str(data)]==cfg['datasets']['PubMed']['sha256']==asset['data_sha256']
   with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
   m=g.StructuralDenoiser(case,cfg,'A').cuda();opt=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
   ema=g.ParameterEMA(m);rng=np.random.default_rng(1)
   g.restore_payload(w.cpu_copy(state),m,opt,ema,binding,lambda v:g.rng_restore(v,rng,'cuda'));m.train();opt.zero_grad(set_to_none=True)
   boundary=g.TrainingBoundary(m,opt,ema,rng,'cuda',2040)
   before=boundary.checkpoint(binding)
   exact={k:g.digest(before[k])==g.digest(state[k]) for k in ['model','optimizer','ema','rng','updates']};assert all(exact.values()),exact
   names=[];params=[]
   for name,param in m.named_parameters():
    if param.requires_grad:names.append(name);params.append(param)
   assert len({id(p) for p in params})==len(params)
   groups=[w.category(n) for n in names];lengths=[p.numel() for p in params]
   slices={};offset=0
   for n,l in zip(names,lengths):slices[n]=(offset,offset+l);offset+=l
   grads={};none={};times={};feedback={}
   completion=read(FULL/'completion.json')
   for draw in [0,1]:
    fp=FULL/f'{tag}_draw{draw}.npz';bind(fp);assert completion['output_hashes'][str(fp)]==BIND[str(fp)]
    with np.load(fp) as z:
     assert np.array_equal(z['slots'],np.arange(len(inp['labels']))) and np.array_equal(z['reference'],inp['labels']) and np.array_equal(z['relation'],inp['f'])
     if draw==0:saved_p=z['probability'].copy()
     else:assert np.array_equal(saved_p,z['probability'])
     for kind in ['old','new']:feedback[f'{kind}_draw{draw}']=z[kind+'_g'].copy()
   with boundary.validation():
    m.eval();token=('parameter_signal',parent,1000);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
     check();torch.cuda.synchronize();tick=time.time()
     graph,logits,p=ctx.predict(inp['query'],inp['f'],1.,token,details=True);status['forwards']+=1
     torch.cuda.synchronize();times['forward_seconds']=time.time()-tick
     g.common.check_prob_device(p,inp['f'],case,m.prior)
     current=p.detach().cpu().double().numpy();atol=cfg['tolerances']['fp32_atol'];rtol=cfg['tolerances']['fp32_rtol']
     mt=dict(parent=parent,checkpoint=str(cp),checkpoint_sha256=expected,input_hash=pr['input_sha256'],updates=2040,
      p_exact=bool(np.array_equal(current,saved_p)),p_max_abs=float(abs(current-saved_p).max()),
      p_nonzero_differences=int(np.count_nonzero(current-saved_p)),within_original_tolerance=bool(np.allclose(current,saved_p,atol=atol,rtol=rtol)),atol=atol,rtol=rtol)
     matching.append(mt);save('probability_matching',matching);assert mt['within_original_tolerance'],'saved probability mismatch; do not widen tolerance'
     assert mt['p_exact'], 'subtraction diagnostic requires exact archived prediction'
     # Actual archived p is expected bitwise under restored strict policy;
     # otherwise frozen vectors are explicitly only local VJP probes.
     labels=torch.as_tensor(inp['labels'],device='cuda');terms=w.supported_binary_ce(logits,labels,m.prior[graph.query_families]);ce=terms.sum()/case['space'].total
     assert torch.isfinite(ce);assert abs(float(ce)-pr['CE'])<=atol+rtol*abs(pr['CE'])
     total_g=(feedback['new_draw0']+feedback['new_draw1'])/2
     common_g=np.zeros_like(total_g)
     rel_rows=[]
     for rel in np.unique(inp['f']):
      sel=inp['f']==rel; common_g[sel]=total_g[sel].mean()
      rel_rows.append(dict(relation=int(rel),slots=int(sel.sum()),mean=float(total_g[sel].mean()),total_norm=float(np.linalg.norm(total_g[sel])),residual_norm=float(np.linalg.norm(total_g[sel]-common_g[sel]))))
     residual_g=total_g-common_g
     assert abs(float(common_g@residual_g))<1e-10+1e-10*float(np.linalg.norm(total_g)**2)
     np.savez_compressed(OUT/f'{parent}_logit_decomposition.npz',total=total_g,common=common_g,residual=residual_g,relation=inp['f'])
     save(parent+'_relation_summary',rel_rows)
     objectives=[('common',((logits[:,1]-logits[:,0])*torch.as_tensor(common_g,device='cuda',dtype=logits.dtype)).sum())]
     rng_probe_before=g.digest(g.rng_capture(rng,'cuda'))
     for j,(name,obj) in enumerate(objectives):
      check();tick=time.time();values=torch.autograd.grad(obj,params,retain_graph=j<len(objectives)-1,allow_unused=True);status['gradient_queries']+=1;torch.cuda.synchronize()
      times[name+'_backward_seconds']=time.time()-tick;none[name]=[n for n,v in zip(names,values) if v is None]
      assert all(v is None or bool(torch.isfinite(v).all()) for v in values)
      grads[name]=torch.cat([(torch.zeros_like(pa) if v is None else v).detach().cpu().reshape(-1) for pa,v in zip(params,values)])
      assert all(pa.grad is None for pa in params),'autograd query polluted main parameter.grad'
      save('progress',dict(**status,parent=parent,last_gradient=name,elapsed_seconds=time.time()-START))
      del values
     assert g.digest(g.rng_capture(rng,'cuda'))==rng_probe_before,'gradient query consumed RNG'
    finally:ctx.close()
   after=boundary.checkpoint(binding)
   unchanged={k:g.digest(before[k])==g.digest(after[k]) for k in ['model','optimizer','ema','rng','updates']};assert all(unchanged.values()),unchanged
   assert m.training is True
   statechecks.append(dict(parent=parent,loaded_exact=exact,after_unchanged=unchanged,model_mode_restored=True,gradient_slots_untouched=True))
   previous=PREV/f'{parent}_parameter_gradients.pt';bind(previous)
   assert read(PREV/'completion.json')['output_hashes'][str(previous)]==BIND[str(previous)]
   previous_payload=torch.load(previous,map_location='cpu',weights_only=False)
   assert previous_payload['checkpoint_sha256']==expected and previous_payload['input_hash']==pr['input_sha256']
   layout=[dict(name=n,group=gr,shape=list(pa.shape),start=slices[n][0],end=slices[n][1]) for n,gr,pa in zip(names,groups,params)]
   assert layout==read(PREV/f'{parent}_parameter_layout.json')
   old=previous_payload['gradients']
   grads['total']=old['new_mean'].double()
   grads['common']=grads['common'].double()
   grads['residual']=grads['total']-grads['common']
   grads['CE']=old['CE'].double()
   save(parent+'_parameter_layout',layout)
   torch.save(dict(gradients=grads,checkpoint_sha256=expected,input_hash=pr['input_sha256'],readout='raw',updates=2040,residual_method='archived total minus common VJP; floating linearity not separately recomputed'),OUT/f'{parent}_decomposition.pt')
   for group in ['all']+sorted(set(groups)):
    ids=[j for j,gr in enumerate(groups) if group=='all' or gr==group]
    gg={key:torch.cat([v[slices[names[j]][0]:slices[names[j]][1]] for j in ids]) for key,v in grads.items()}
    total,com,res=gg['total'],gg['common'],gg['residual'];tn=float(total.norm())
    summary.append(dict(parent=parent,group=group,total_norm=tn,common_norm=float(com.norm()),residual_norm=float(res.norm()),
      common_over_total=float(com.norm())/tn if tn else None,residual_over_total=float(res.norm())/tn if tn else None,
      common_projection_fraction=float(com@total)/(tn*tn) if tn else None,residual_projection_fraction=float(res@total)/(tn*tn) if tn else None,
      common_total_cosine=compare(com,total)['cosine'],residual_total_cosine=compare(res,total)['cosine'],common_residual_cosine=compare(com,res)['cosine'],
      common_CE_cosine=compare(com,gg['CE'])['cosine'],residual_CE_cosine=compare(res,gg['CE'])['cosine'],
      norm_identity_error=abs(float(total@total-com@com-res@res-2*(com@res)))))
   for n in names:
    lo,hi=slices[n]
    byparam.append(dict(parent=parent,name=n,group=w.category(n),numel=hi-lo,total_norm=float(grads['total'][lo:hi].norm()),common_norm=float(grads['common'][lo:hi].norm()),residual_norm=float(grads['residual'][lo:hi].norm())))
   save('comparisons',summary);save('per_parameter',byparam);save('state_checks',statechecks);save(parent+'_timings',times)
   status['completed'].append(parent);print(json.dumps(dict(completed=parent,matching=mt,counts=status)),flush=True)
   del ctx,graph,logits,p,objectives,terms,ce,obj,params,m,opt,ema,boundary,before,after,state,env,grads,gg
   gc.collect();torch.cuda.empty_cache();check()
 fr.authenticate_sources();assert status['forwards']==2 and status['gradient_queries']==2 and status['completed']==['CE','PS']
 assert all(w.sha(p)==h for p,h in BIND.items());save('binding',BIND);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(worker_seconds=time.time()-START,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0.,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0.)
 save('receipt',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
