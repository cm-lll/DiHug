"""DIRECT hard-score parameter gradient probe; four forwards/eight autograd, zero updates."""
import os,sys,json,time,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ['OBJECTIVE_DIAG_OUT'])
spec=importlib.util.spec_from_file_location('helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,fr,torch,np=w.g,w.fr,w.torch,w.np
sys.path.insert(0,str(ROOT/'research_branches/reproducible_aggregation_20261108'))
from adapter import execution_policy,VERSION
sys.path.insert(0,str(ROOT/'research_branches/direct_structure_cpu_20261108'))
from score import DirectScore
OLD=ROOT/'results/reproducible_objective_20261108T135622/attempt2';REF=ROOT/'results/update_response_20261108T142622'
P=w.read(OUT/'protocol.json');start=time.time();status=dict(forwards=0,backwards=0,updates=0,status='running');results=[];assets=[]
def save(n,x):fr.save(OUT/n,x)
def check():
 if time.time()>=P['science_stop_epoch']-10:raise TimeoutError('science cutoff')
def norm(x):return float(x.double().norm())
def flatten(v,params):return torch.cat([(torch.zeros_like(p) if x is None else x).detach().cpu().flatten() for p,x in zip(params,v)])
def main():
 fr.setup();fr.authenticate_sources();assert torch.cuda.is_available() and torch.cuda.device_count()==1
 for path,sha in w.read(OUT/'source_binding.json').items():assert w.sha(path)==sha
 save('environment_before.json',g.common.environment('cuda'))
 artifacts=w.read(OLD/'audit.json')['local_artifacts']
 for task in w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json'):
  ds,cfg=task['dataset'],task['row']['config'];check()
  data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets'][ds]['file'];assert w.sha(data)==cfg['datasets'][ds]['sha256']
  with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
  history=[json.loads(x) for x in (w.OLD/'run'/task['task']/'training.jsonl').read_text().splitlines()]
  m=g.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=g.ParameterEMA(m);rng=np.random.default_rng(1)
  named=list(m.named_parameters());params=[p for _,p in named]
  with execution_policy(True):
   save('environment_active.json',dict(**g.common.environment('cuda'),execution_policy=VERSION))
   for arm in ['CE','PS']:
    check();condition=ds+'_'+arm;cp=OLD/f'{condition}_diagnostic_state.pt';sha=w.sha(cp);assert sha==artifacts[str(cp.relative_to(ROOT))]
    envelope=torch.load(cp,map_location='cpu');assert envelope['diagnostic_only'] and envelope['local_objective']==arm and envelope['execution_policy']==VERSION
    parent=envelope['state'];binding=parent['binding'];assert binding['config']==cfg and parent['updates']==task['start']+32
    g.restore_payload(w.cpu_copy(parent),m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'));m.train();o.zero_grad(set_to_none=True)
    boundary=g.TrainingBoundary(m,o,e,rng,'cuda',parent['updates'])
    before=boundary.checkpoint(binding)
    for k in ['model','optimizer','ema','rng','updates']:assert g.digest(before[k])==g.digest(parent[k]),k
    u=parent['updates']+1;inp=g.one_input(case,cfg,u,rng);ih=w.input_hash([inp]);assert ih==history[u-1]['input_sha256']
    post_input=g.digest(g.rng_capture(rng,'cuda'));assert post_input==history[u-1]['full_rng_before']
    refpath=REF/f'{condition}_local_tensors.pt';old=torch.load(refpath,map_location='cpu');assert old['parameter_names']==[n for n,p in named]
    tick=time.perf_counter();token=('direct_structure_gradient',condition,u);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
     graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);status['forwards']+=1;torch.cuda.synchronize();forward_s=time.perf_counter()-tick
     g.common.check_prob_device(p,inp['f'],case,m.prior)
     ce=w.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/case['space'].total
     tick=time.perf_counter();cg=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True);status['backwards']+=1;torch.cuda.synchronize();ce_backward_s=time.perf_counter()-tick
     cflat=flatten(cg,params);err=float((cflat-old['actual_gradients']['CE']).abs().max());assert torch.allclose(cflat,old['actual_gradients']['CE'],atol=2e-6,rtol=1e-5),'CE reference mismatch'
     prob=p.detach().cpu().double().numpy();tick=time.perf_counter();aux_rng=np.random.default_rng(P['aux_seeds'][condition]);bits=(aux_rng.random(len(prob))<prob).astype(np.float64)
     direct=DirectScore(case['space'].types,inp['query'],inp['f'],inp['labels'],bits,case['n_families']);outg=direct.logit_estimate(prob);cpu_s=time.perf_counter()-tick
     assert np.isfinite(outg).all();margin=logits[:,1]-logits[:,0]
     # Independent edge Bernoulli p0. Detached exact edge-conditional estimator is a margin gradient, NOT a loss weight.
     active=(prob>0)&(prob<1);prob_error=float(np.max(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()[active]-prob[active]))) if active.any() else 0.
     assert prob_error<2e-6,'probability/logit derivative contract'
     gradout=torch.as_tensor(outg,device='cuda',dtype=margin.dtype);tick=time.perf_counter();sg=torch.autograd.grad(margin,params,grad_outputs=gradout,allow_unused=True);status['backwards']+=1;torch.cuda.synchronize();structure_backward_s=time.perf_counter()-tick
     sflat=flatten(sg,params);assert torch.isfinite(cflat).all() and torch.isfinite(sflat).all()
     assert all(p.grad is None for p in params)
    finally:ctx.close()
    after=boundary.checkpoint(binding)
    for k in ['model','optimizer','ema','updates']:assert g.digest(after[k])==g.digest(before[k]),'state polluted '+k
    assert g.digest(g.rng_capture(rng,'cuda'))==post_input,'training RNG polluted'
    geometry={}
    for group in ['all','CN_exclusive','joint_head','degree_branch','shared_backbone_other']:
     ix=[(lo,hi) for name,c,lo,hi in old['spans'] if group=='all' or c==group];cv=torch.cat([cflat[lo:hi] for lo,hi in ix]);sv=torch.cat([sflat[lo:hi] for lo,hi in ix]);cn,sn=norm(cv),norm(sv)
     geometry[group]=dict(CE_L2=cn,structure_L2=sn,ratio=sn/cn if cn else None,cosine=float(cv.double()@sv.double())/(cn*sn) if cn*sn else None)
    row=dict(condition=condition,seed=task['seed'],parent_updates=parent['updates'],training_t=inp['t'],input_sha256=ih,CE=float(ce),CE_reference_max_abs=err,probability_contract_max_abs=prob_error,aux_seed=P['aux_seeds'][condition],aux_draws=1,geometry=geometry,timing=dict(forward=forward_s,CE_backward=ce_backward_s,CPU_structure=cpu_s,structure_backward=structure_backward_s),CE_none=[n for (n,p),x in zip(named,cg) if x is None],structure_none=[n for (n,p),x in zip(named,sg) if x is None])
    np.savez_compressed(OUT/f'{condition}_output_gradient.npz',gradient=outg,probability=prob)
    torch.save(dict(parameter_names=[n for n,p in named],spans=old['spans'],CE=cflat,structure=sflat),OUT/f'{condition}_parameter_gradients.pt')
    g.restore_payload(w.cpu_copy(parent),m,o,e,binding,lambda x:g.rng_restore(x,rng,'cuda'))
    restored=boundary.checkpoint(binding)
    for k in ['model','optimizer','ema','rng','updates']:assert g.digest(restored[k])==g.digest(parent[k]),'final restore '+k
    row['full_restore_equal']=True
    mmdpath=ROOT/'results/hard_structure_gradient_gpu_20261108T180629'/f'{condition}_parameter_gradients.pt';mmd=torch.load(mmdpath,map_location='cpu');assert mmd['parameter_names']==[n for n,p in named];assert torch.equal(mmd['CE'],cflat),'historical CE gradient changed'
    with np.load(mmdpath.with_name(f'{condition}_output_gradient.npz')) as z:assert np.array_equal(z['probability'],prob),'historical probability changed'
    oldstruct=mmd['structure'];sn,mn=norm(sflat),norm(oldstruct);row['DIRECT_vs_MMD']=dict(norm_ratio=sn/mn if mn else None,cosine=float(sflat.double()@oldstruct.double())/(sn*mn) if sn*mn else None,reference=str(mmdpath),sha256=w.sha(mmdpath));row['historical_probability_and_CE_exact']=True
    results.append(row);save('results.json',results);del mmd,oldstruct
    assets.append(dict(condition=condition,parent=str(cp),sha256=sha,reference=str(refpath),reference_sha256=w.sha(refpath),data_sha256=w.sha(data),input_sha256=ih));save('assets.json',assets)
    assert w.sha(cp)==sha
    print(json.dumps(dict(condition=condition,geometry=geometry['all'],timing=row['timing'])),flush=True)
    del old,parent,envelope,before,after,restored,cg,sg,cflat,sflat,direct,graph,logits,p,ce,margin
  del m,o,e,boundary;torch.cuda.empty_cache()
 fr.authenticate_sources();assert (status['forwards'],status['backwards'],status['updates'])==(4,8,0);status['status']='complete'
try:main()
except BaseException as exc:status.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
finally:
 status.update(seconds=time.time()-start,end_epoch=time.time(),peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,completed=len(results));save('receipt.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['status']=='complete' else 2)
