"""Three matched temporary updates, frozen increment heat reward; no sampling.
Derived from the prior bounded update check. Does not alter production sources.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,csv,traceback,importlib.util,subprocess,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/increment_update_check_20261113'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
START=1794545159.0468729
STOP=START+1320
DELIVERY=START+1500
GPU_LIMIT_SECONDS=1080
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
 launch_receipt=json.loads((OUT/'launch.json').read_text())
 actual_stop=min(STOP,launch_receipt['allocation_start']+GPU_LIMIT_SECONDS)
 def check():
  if time.time()>=actual_stop: raise TimeoutError('fixed science/card-time deadline')
 def group(n):
  return 'CN' if n.startswith('common_neighbor.') else 'joint_head' if n.startswith('joint_head.') else 'degree' if n.startswith('branch.') else 'backbone'
 try:
  p=json.loads((PANEL/'protocol.json').read_text())
  for path,h in p['bindings'].items():
   assert s.base.sha(path)==h,path
  scorepath=ROOT/'results/increment_reward_audit_20261113_v2/graphs.csv'
  scores=list(csv.DictReader(scorepath.open()));sources[str(scorepath)]=s.base.sha(scorepath)
  write('environment.json',w.common.environment('cuda'))
  frozen=json.loads((OUT/'preflight.json').read_text())
  for asset,h in frozen['bindings'].items():assert s.base.sha(asset)==h,asset
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
    rr=[];prob_dump={};before=s.state_digest(boundary.checkpoint(state['binding']))
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
      from scipy.spatial.distance import cdist
      from scipy.optimize import linear_sum_assignment
      anonymous=[]
      for ty in sorted(set(case['space'].types.tolist())):
       ix=case['space'].types==ty;active=ref[ix].sum(0)>0;dr=pred[ix].sum(1);rd=ref[ix].sum(1)
       cost=cdist(pred[ix][:,active],ref[ix][:,active],metric='cityblock')/int(active.sum());ii,jj=linear_sum_assignment(cost)
       top=max(1,int(np.ceil(.1*int(ix.sum()))))
       anonymous.append(dict(type=int(ty),nodes=int(ix.sum()),total_degree_W1=float(abs(np.sort(dr)-np.sort(rd)).mean()),
        joint_relation_degree_L1=float(cost[ii,jj].mean()),marginal_relation_degree_W1=float(abs(np.sort(pred[ix][:,active],axis=0)-np.sort(ref[ix][:,active],axis=0)).mean()),
        top10_degree_mean_error=abs(float(np.sort(dr)[-top:].mean()-np.sort(rd)[-top:].mean()))))
      prob_dump['p_'+str(inp['t'])]=pp;prob_dump['query_'+str(inp['t'])]=q;prob_dump['family_'+str(inp['t'])]=f;prob_dump['label_'+str(inp['t'])]=lab
      rr.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=float(term.sum()/case['space'].total),node_total_degree_MAE=float(dg.mean()),relations=rel,type_groups=types,anonymous=anonymous,
       expected_edges=float(pp.sum()),reference_edges=int(lab.sum()),expected_edge_error=abs(float(pp.sum())-int(lab.sum())),
       positive_CE=float(term[lab==1].mean()),negative_CE=float(term[lab==0].mean())))
    tag='baseline' if boundary.completed==0 else arm
    np.savez_compressed(OUT/f'{job["id"]}_{tag}_probabilities.npz',**prob_dump,types=case['space'].types)
    assert before==s.state_digest(boundary.checkpoint(state['binding']))
    return rr
   result=dict(job=job['id'],dataset=ds,seed=seed,baseline=evaluate(),arms={},serialized_restore_exact=True,CUDA_recomputation_bitwise_certified=False)
   prior=json.loads((ROOT/'results/robust_heat_update_check_20261113T095140'/(job['id']+'.json')).read_text())
   assert [r['input_hash'] for r in result['baseline']]==[r['input_hash'] for r in prior['baseline']]
   result['baseline_log_CE_max_difference']=max(abs(a['CE']-b['CE']) for a,b in zip(result['baseline'],prior['baseline']))
   initial={n:v.detach().cpu().clone() for n,v in m.named_parameters()};vectors={};grads={}
   batch=json.loads((directory/'round1_batch.json').read_text())
   for arm in ['original','increment','amplitude_control']:
    boundary=restore();o.zero_grad(set_to_none=True)
    costs=[]
    for plan in job['terminal_plan']['1']:
     found=[r for r in scores if r['dataset']==ds and int(r['train_seed'])==seed and r['arm']=='A' and int(r['update'])==1 and int(r['chain_seed'])==plan['seed']]
     assert len(found)==1;costs.append(float(found[0][arm+'_score']))
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
       assert torch.isfinite(loss)
       loss.backward();backs+=1
     assert rh==s.state_digest(s.rng_capture(rng,'cuda'))
     grads[arm]={n:v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu') for n,v in m.named_parameters()}
     assert all(torch.isfinite(v).all() for v in grads[arm].values())
     norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1,error_if_nonfinite=True))
     boundary.mark_optimizer_started();o.step();e.update(m,1);e.at_update_boundary=True
     assert all(torch.isfinite(v).all() for v in m.parameters())
    vectors[arm]={n:v.detach().cpu()-initial[n] for n,v in m.named_parameters()}
    complete=boundary.checkpoint(state['binding']);assert complete['updates']==1 and e.updates==1
    cp=OUT/f'{job["id"]}_{arm}_temporary.pt';torch.save(complete,cp)
    loaded=torch.load(cp,map_location='cpu',weights_only=False);assert s.state_digest(loaded)==s.state_digest(complete)
    s.restore_payload(loaded,m,o,e,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'));m.eval()
    assert s.state_digest(boundary.checkpoint(state['binding']))==s.state_digest(complete)
    assert s.state_digest(complete['rng'])==s.state_digest(state['rng'])
    result['arms'][arm]=dict(costs=costs.tolist(),advantages=adv.tolist(),gradient_norm_before_clip=norm,clip_scale=min(1,1/(norm+1e-6)),update_seconds=time.time()-tick,validation=evaluate(),complete_state_sha256=s.base.sha(cp),EMA_updates=e.updates,RNG_preserved=True)
    write(job['id']+'.json',result);write('progress.json',dict(job=job['id'],arm=arm,forwards=calls,backwards=backs,completed=len(done)))
   comparisons=[]
   for kind,vs in [('gradient',grads),('actual_update',vectors)]:
    for left,right in [('original','increment'),('original','amplitude_control'),('amplitude_control','increment')]:
     for g in ['ALL','backbone','CN','degree','joint_head']:
      names=[n for n in initial if g=='ALL' or group(n)==g]
      aa=sum(float(vs[left][n].double().square().sum()) for n in names);bb=sum(float(vs[right][n].double().square().sum()) for n in names)
      dot=sum(float((vs[left][n].double()*vs[right][n].double()).sum()) for n in names)
      comparisons.append(dict(kind=kind,group=g,left=left,right=right,left_norm=aa**.5,right_norm=bb**.5,cosine=dot/max((aa*bb)**.5,1e-300)))
   result['comparisons']=comparisons;write(job['id']+'.json',result);done.append(job['id'])
   del m,o,e,state,parent,initial,vectors,grads,complete,loaded;torch.cuda.empty_cache()
  assert calls==480 and backs==384
  assert all(s.base.sha(path)==h for path,h in sources.items())
  write('completion.json',dict(status='complete',conditions=done,forwards=calls,backwards=backs,new_graphs=0,formal_training=False,end_epoch=time.time(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20))
 except BaseException:
  write('failure.json',dict(error=traceback.format_exc(),conditions=done,forwards=calls,backwards=backs,end_epoch=time.time()));raise
 finally:write('sources.json',sources)
def prepare():
 if OUT.exists():raise RuntimeError('Output directory already exists, do not duplicate run')
 OUT.mkdir()
 proto=dict(start_epoch=START,science_stop_epoch=STOP,delivery_epoch=DELIVERY,max_GPU_hours=.3,GPU_limit_seconds=GPU_LIMIT_SECONDS,GPU=0,
  conditions=['IMDB0411','IMDB0412','PubMed0421','PubMed0422'],arms=['original','increment','amplitude_control'],
  temporary_updates=12,saved_trajectories=16,saved_Xt=128,forward_backward_calls=384,raw_validation_calls=96,
  total_forwards=480,times=[1,75,250,550,800,1000],validation_seed_root=202611130953,
  readout='raw; EMA state checked, not EMA quality',validation='previously viewed fixed forward-noise panel, not independent confirmation',
  rule='frozen heat_increment_cpu candidate and frozen amplitude control; no tuning',new_graphs=0,formal_training=False,
  state='four original feedback step0 complete states, model from parent EMA3000, original empty feedback AdamW retained',
  numeric='retain prior deterministic flags and precision; CUDA repeated-update tolerance still not certified',
  budget_basis='previous 8 temporary updates:328F/256B,456.22GPU seconds; new12:480F/384B nominal~680s; cap1080GPU seconds; CPU save/load included')
 write('protocol.json',proto)
 p=json.loads((PANEL/'protocol.json').read_text());bindings=dict(p['bindings'])
 sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 assert all(sha(path)==h for path,h in bindings.items())
 for f in [Path(__file__),ROOT/'results/increment_reward_audit_20261113_v2/graphs.csv',ROOT/'results/heat_increment_cpu_20261113/control_frozen.json',
           ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py']:
  bindings[str(f)]=sha(f)
 jobs=[]
 for job in p['jobs']:
  if job['architecture']!='A':continue
  row=p['parents'][job['parent']];ds=job['dataset'];directory=PANEL/job['id']/(ds+'_T_eager')
  files=[directory/'initial_complete.pt',Path(row['readout']),Path(row['data']),directory/'round1_batch.json']
  assert len(job['terminal_plan']['1'])==4
  for plan in job['terminal_plan']['1']:
   folder=directory/f'train_round1_seed{plan["seed"]}'
   assert len(plan['times'])==8
   files.extend([folder/'graph.npz',folder/'identity.json',folder/'summary.json'])
   files.extend(folder/f'Xt_{t:04d}.npz' for t in plan['times'])
  for f in files:
   assert f.is_file(),f;bindings[str(f)]=sha(f)
  jobs.append(job['id'])
 assert len(jobs)==4
 write('preflight.json',dict(status='assets_hashes_ready',bindings=bindings,jobs=jobs,all_cached_inputs=128,GPU_calls=0,
  GPU_prelaunch_required=True,disk='/data2 460GiB free; /174GiB free at read-only resource inspection'))
 print(json.dumps(proto,indent=2))
def launch():
 if (OUT/'launch.json').exists():raise RuntimeError('Duplicate launch refused')
 p=json.loads((OUT/'preflight.json').read_text())
 sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 assert all(sha(path)==h for path,h in p['bindings'].items())
 if time.time()>STOP-850:raise RuntimeError('Insufficient time for full 12-update panel')
 q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True,timeout=30)
 rows=[r.strip().split(', ') for r in q.stdout.strip().splitlines()];gpu=next(r for r in rows if r[0]=='0')
 if int(gpu[2])!=0 or int(gpu[3])>5:raise RuntimeError('GPU0 no longer idle, stop without taking resource')
 start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 write('launch.json',dict(status='allocating',GPU=0,uuid=gpu[1],allocation_start=start,science_stop_epoch=STOP,delivery_epoch=DELIVERY,query=q.stdout))
 with (OUT/'worker.log').open('w') as f:
  child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  receipt=json.loads((OUT/'launch.json').read_text());receipt.update(pid=child.pid,status='running');write('launch.json',receipt)
  try:
   code=child.wait(timeout=max(.1,min(STOP-start,GPU_LIMIT_SECONDS-5)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=child.returncode
  finally:
   end=time.time();write('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,
     allocation_start=start,end_epoch=end,GPU_hours=(end-start)/3600))
 if code:sys.exit(code)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
 elif len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 elif len(sys.argv)>1 and sys.argv[1]=='launch':launch()
 else:raise SystemExit('prepare or launch explicitly required')

