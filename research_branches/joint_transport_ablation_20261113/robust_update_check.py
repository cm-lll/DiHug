"""Bounded paired diagnostic; no sampling or production continuation."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,csv,traceback,importlib.util,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/robust_heat_update_check_20261113T095140'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
START=1794534700.0  # overwritten below from explicit timezone, never from process restart
from datetime import datetime
START=datetime.fromisoformat('2026-11-13T09:51:40+08:00').timestamp()
STOP=START+810
def write(name,obj):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def worker():
 spec=importlib.util.spec_from_file_location('check_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
 s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 w,np,torch=s.w,s.np,s.torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 calls=backs=0;done=[];sources={}
 def check():
  if time.time()>=STOP: raise TimeoutError('fixed science deadline')
 def group(n):
  return 'CN' if n.startswith('common_neighbor.') else 'joint_head' if n.startswith('joint_head.') else 'degree' if n.startswith('branch.') else 'backbone'
 try:
  p=json.loads((PANEL/'protocol.json').read_text())
  for path,h in p['bindings'].items():
   assert s.base.sha(path)==h,path
  scorepath=ROOT/'results/robust_heat_transport_20261113/training_rescore.csv'
  scores=list(csv.DictReader(scorepath.open()));sources[str(scorepath)]=s.base.sha(scorepath)
  write('environment.json',w.common.environment('cuda'))
  for job in p['jobs']:
   if job['architecture']!='A':continue
   check();row=p['parents'][job['parent']];ds=job['dataset'];seed=row['train_seed']
   directory=PANEL/job['id']/f'{ds}_T_eager'
   path=directory/'initial_complete.pt';state=torch.load(path,map_location='cpu',weights_only=False)
   parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
   assert state['updates']==0 and not state['optimizer']['state']
   assert s.state_digest(state['model'])==s.state_digest(parent['model'])
   for asset in (path,Path(row['readout']),Path(row['data'])):sources[str(asset)]=s.base.sha(asset)
   z=np.load(row['data']);ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
   case=w.common.make_case(z['types'],ee,tuple(map(tuple,z['schema'])))
   m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval()
   oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
   o=torch.optim.AdamW(m.parameters(),**oc);e=s.ParameterEMA(m,.99);rng=np.random.default_rng()
   def restore():
    s.restore_payload(state,m,o,e,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
    b=s.TrainingBoundary(m,o,e,rng,'cuda',0)
    assert s.state_digest(b.checkpoint(state['binding']))==s.state_digest(state)
    m.eval();return b
   boundary=restore();ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
   labels=np.isin(ids,case['clean_ids']).astype(np.uint8)
   fixed=[w.common.train_input(case,1000,1,seed,920000+j,np.random.default_rng(np.random.SeedSequence([202611130953,seed,j])),force_t=t) for j,t in enumerate([1,75,250,550,800,1000])]
   def evaluate():
    nonlocal calls
    rr=[];before=s.state_digest(boundary.checkpoint(state['binding']))
    with torch.no_grad():
     for inp in fixed:
      check();logits,prob=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
      w.common.check_prob_device(prob,inp['f'],case,m.prior)
      term=s.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy()
      pp=prob.cpu().double().numpy();q=inp['query'];f=inp['f'];lab=inp['labels'];n=len(case['space'].types)
      pred=np.zeros((n,case['n_families']));ref=np.zeros_like(pred)
      for endpoint in q:
       np.add.at(pred,(endpoint,f),pp);np.add.at(ref,(endpoint,f),lab)
      rel=[]
      for r in range(case['n_families']):
       pos=(f==r)&(lab==1);neg=(f==r)&(lab==0)
       rel.append(dict(family=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(term[pos].mean()) if pos.any() else None,negative_CE=float(term[neg].mean()) if neg.any() else None,expected_edges=float(pp[f==r].sum()),reference_edges=int(pos.sum()),missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum()),node_relation_degree_MAE=float(abs(pred[:,r]-ref[:,r]).mean())))
      dg=abs(pred.sum(1)-ref.sum(1));types=[]
      for ty in sorted(set(case['space'].types.tolist())):
       ix=case['space'].types==ty;rd=ref.sum(1);hub=ix&(rd>=np.quantile(rd[ix],.9));low=ix&(rd<=1)
       types.append(dict(type=int(ty),low_n=int(low.sum()),hub_n=int(hub.sum()),low_MAE=float(dg[low].mean()) if low.any() else None,hub_MAE=float(dg[hub].mean()) if hub.any() else None))
      rr.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=float(term.sum()/case['space'].total),node_total_degree_MAE=float(dg.mean()),relations=rel,type_groups=types))
    assert before==s.state_digest(boundary.checkpoint(state['binding']))
    return rr
   result=dict(job=job['id'],dataset=ds,seed=seed,baseline=evaluate(),arms={},serialized_restore_exact=True,CUDA_recomputation_bitwise_certified=False)
   initial={n:v.detach().cpu().clone() for n,v in m.named_parameters()};vectors={};grads={}
   batch=json.loads((directory/'round1_batch.json').read_text())
   for arm in ['original','robust']:
    boundary=restore();o.zero_grad(set_to_none=True)
    costs=[]
    for plan in job['terminal_plan']['1']:
     found=[r for r in scores if r['dataset']==ds and int(r['train_seed'])==seed and r['arm']=='A' and int(r['update'])==1 and int(r['chain_seed'])==plan['seed']]
     assert len(found)==1;costs.append(float(found[0][arm]))
    costs=np.array(costs);adv=(4*costs-costs.sum())/3
    if arm=='original':assert np.allclose(costs,batch['costs'],rtol=0,atol=1e-8)
    tick=time.time()
    with boundary.step(1):
     rh=s.state_digest(s.rng_capture(rng,'cuda'))
     for k,plan in enumerate(job['terminal_plan']['1']):
      folder=directory/f'train_round1_seed{plan["seed"]}';identity=json.loads((folder/'identity.json').read_text())
      assert identity['model_hash']==s.state_digest(state['model']) and identity['updates']==0
      graph=np.load(folder/'graph.npz');ed=graph['edges'];ed=ed if ed.shape[0]==2 else ed.T
      eset={tuple(sorted(map(int,x))) for x in ed.T};bits=np.array([tuple(sorted(map(int,x))) in eset for x in query.T]);assert bits.sum()==len(eset)
      target=torch.as_tensor(bits,device='cuda').bool()
      for t in sorted(plan['times'],reverse=True):
       check();xp=folder/f'Xt_{t:04d}.npz';sources[str(xp)]=s.base.sha(xp);zz=np.load(xp);assert int(zz['t'])==t
       ed,ff=w.common.state_edges(case,zz['xt'])
       logits,prob=w.runtime.predict(m,case['space'].types,ed,ff,query,fam,t/1000,details=True);calls+=1
       w.common.check_prob_device(prob,fam,case,m.prior)
       lp=torch.log_softmax(logits.double(),dim=-1)
       loss=.01*float(adv[k])*torch.where(target,lp[:,1],lp[:,0]).sum()/32
       loss.backward();backs+=1
     assert rh==s.state_digest(s.rng_capture(rng,'cuda'))
     grads[arm]={n:v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu') for n,v in m.named_parameters()}
     norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1,error_if_nonfinite=True))
     boundary.mark_optimizer_started();o.step();e.update(m,1);e.at_update_boundary=True
     assert all(torch.isfinite(v).all() for v in m.parameters())
    vectors[arm]={n:v.detach().cpu()-initial[n] for n,v in m.named_parameters()}
    complete=boundary.checkpoint(state['binding']);assert complete['updates']==1 and e.updates==1
    cp=OUT/f'{job["id"]}_{arm}_temporary.pt';torch.save(complete,cp)
    loaded=torch.load(cp,map_location='cpu',weights_only=False);assert s.state_digest(loaded)==s.state_digest(complete)
    result['arms'][arm]=dict(costs=costs.tolist(),advantages=adv.tolist(),gradient_norm_before_clip=norm,clip_scale=min(1,1/(norm+1e-6)),update_seconds=time.time()-tick,validation=evaluate(),complete_state_sha256=s.base.sha(cp),EMA_updates=e.updates,RNG_preserved=True)
    write(job['id']+'.json',result);write('progress.json',dict(job=job['id'],arm=arm,forwards=calls,backwards=backs,completed=len(done)))
   comparisons=[]
   for kind,vs in [('gradient',grads),('actual_update',vectors)]:
    for g in ['ALL','backbone','CN','degree','joint_head']:
     names=[n for n in initial if g=='ALL' or group(n)==g]
     a=sum(float(vs['original'][n].double().square().sum()) for n in names);b=sum(float(vs['robust'][n].double().square().sum()) for n in names)
     dot=sum(float((vs['original'][n].double()*vs['robust'][n].double()).sum()) for n in names)
     comparisons.append(dict(kind=kind,group=g,original_norm=a**.5,robust_norm=b**.5,cosine=dot/max((a*b)**.5,1e-300)))
   result['comparisons']=comparisons;write(job['id']+'.json',result);done.append(job['id'])
   del m,o,e,state,parent,initial,vectors,grads,complete,loaded;torch.cuda.empty_cache()
  write('completion.json',dict(status='complete',conditions=done,forwards=calls,backwards=backs,new_graphs=0,formal_training=False,end_epoch=time.time(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20))
 except BaseException:
  write('failure.json',dict(error=traceback.format_exc(),conditions=done,forwards=calls,backwards=backs,end_epoch=time.time()));raise
 finally:write('sources.json',sources)
def launch():
 OUT.mkdir(parents=True,exist_ok=True)
 write('protocol.json',dict(start_epoch=START,science_stop_epoch=STOP,delivery_epoch=START+900,max_GPU_hours=.25,GPU=0,conditions=['IMDB0411','IMDB0412','PubMed0421','PubMed0422'],arms=['original','robust'],temporary_updates=8,saved_trajectories=16,saved_Xt=128,forward_backward_calls=256,independent_raw_validation_calls=72,times=[1,75,250,550,800,1000],validation_seed_root=202611130953,readout='raw; EMA state only, not EMA quality',rule='fixed robust_heat_v1, no tuning',new_graphs=0))
 if time.time()>STOP-300:raise RuntimeError('Insufficient remaining window to start complete panel')
 start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with (OUT/'worker.log').open('w') as f:
  child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write('launch.json',dict(pid=child.pid,GPU=0,allocation_start=start,science_stop_epoch=STOP,delivery_epoch=START+900))
  try:code=child.wait(timeout=max(.1,min(STOP-start,900)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode
  finally:write('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,allocation_start=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600))
 if code:sys.exit(code)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:launch()
