"""Fixed last-half-attention diagnostic; isolated, no new sampling chains."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,re,json,csv,time,traceback,importlib.util,subprocess,signal,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/partial_attention_check_20261113T101200'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
PREV=ROOT/'results/robust_heat_update_check_20261113T095140'
START=datetime.fromisoformat('2026-11-13T10:12:00+08:00').timestamp()
STOP=START+810
def write(name,x):
 OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def selected(name):return re.match(r'^backbone\.tf_layers\.[4-7]\.self_attn\.',name) is not None
def apply_mask(model):
 layers={int(n.split('tf_layers.')[1].split('.')[0]) for n,_ in model.named_parameters() if 'tf_layers.' in n}
 assert layers==set(range(8)),layers
 names=[]
 for n,p in model.named_parameters():
  p.grad=None;p.requires_grad_(selected(n))
  if p.requires_grad:names.append(n)
 assert names and all('.self_attn.' in n for n in names)
 return names
def worker():
 spec=importlib.util.spec_from_file_location('partial_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
 s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);w,np,torch=s.w,s.np,s.torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 existing=[json.loads(f.read_text()) for f in sorted(OUT.glob('*_A_transport.json'))]
 assert all(x['restore_exact'] and x['frozen_raw_exact'] and x['RNG_unchanged'] for x in existing)
 done=[x['job'] for x in existing];calls=38*len(done);backs=32*len(done)
 sources=json.loads((OUT/'sources.json').read_text()) if (OUT/'sources.json').exists() else {}
 def check():
  if time.time()>=STOP:raise TimeoutError('absolute scientific deadline')
 try:
  protocol=json.loads((PANEL/'protocol.json').read_text())
  for path,h in protocol['bindings'].items():assert s.base.sha(path)==h,path
  rows=list(csv.DictReader((ROOT/'results/robust_heat_transport_20261113/training_rescore.csv').open()))
  write('environment.json',w.common.environment('cuda'))
  for job in protocol['jobs']:
   if job['architecture']!='A':continue
   if job['id'] in done:continue
   check();row=protocol['parents'][job['parent']];ds=job['dataset'];seed=row['train_seed'];folder=PANEL/job['id']/f'{ds}_T_eager'
   old=json.loads((PREV/(job['id']+'.json')).read_text());assert set(old['arms'])=={'original','robust'}
   state=torch.load(folder/'initial_complete.pt',map_location='cpu',weights_only=False)
   parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
   assert state['updates']==0 and not state['optimizer']['state'] and s.state_digest(state['model'])==s.state_digest(parent['model'])
   z=np.load(row['data']);ed=z['edges'];ed=ed if ed.shape[0]==2 else ed.T
   case=w.common.make_case(z['types'],ed,tuple(map(tuple,z['schema'])))
   m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval();oc={k:v for k,v in cfg['optimizer'].items() if k!='type'};oc['lr']=1e-5
   full_opt=torch.optim.AdamW(m.parameters(),**oc);e=s.ParameterEMA(m,.99);rng=np.random.default_rng()
   s.restore_payload(state,m,full_opt,e,state['binding'],lambda x:s.rng_restore(x,rng,'cuda'))
   assert s.state_digest(s.TrainingBoundary(m,full_opt,e,rng,'cuda',0).checkpoint(state['binding']))==s.state_digest(state)
   names=apply_mask(m);params=[p for p in m.parameters() if p.requires_grad];o=torch.optim.AdamW(params,**oc)
   old_options={k:v for k,v in full_opt.state_dict()['param_groups'][0].items() if k!='params'}
   assert old_options=={k:v for k,v in o.state_dict()['param_groups'][0].items() if k!='params'};del full_opt
   binding=dict(state['binding'],diagnostic='last_half_self_attention_only',trainable_names=names,source_initial_sha=s.base.sha(folder/'initial_complete.pt'))
   boundary=s.TrainingBoundary(m,o,e,rng,'cuda',0);initial=boundary.checkpoint(binding)
   cp=OUT/f'{job["id"]}_initial.pt';torch.save(initial,cp);loaded=torch.load(cp,map_location='cpu',weights_only=False)
   s.restore_payload(loaded,m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'))
   assert s.state_digest(boundary.checkpoint(binding))==s.state_digest(initial)
   assert [n for n,p in m.named_parameters() if p.requires_grad]==names
   ids,fam,query=w.common.plan_for(case,2026111402,1).block(0)
   fixed=[w.common.train_input(case,1000,1,seed,920000+j,np.random.default_rng(np.random.SeedSequence([202611130953,seed,j])),force_t=t) for j,t in enumerate([1,75,250,550,800,1000])]
   assert [s.input_hash([x]) for x in fixed]==[r['input_hash'] for r in old['baseline']]
   costs=[]
   for plan in job['terminal_plan']['1']:
    rr=[r for r in rows if r['dataset']==ds and int(r['train_seed'])==seed and r['arm']=='A' and int(r['update'])==1 and int(r['chain_seed'])==plan['seed']]
    assert len(rr)==1;costs.append(float(rr[0]['robust']))
   costs=np.array(costs);adv=(4*costs-costs.sum())/3
   assert np.array_equal(costs,np.array(old['arms']['robust']['costs']))
   assert np.array_equal(adv,np.array(old['arms']['robust']['advantages']))
   before={n:p.detach().cpu().clone() for n,p in m.named_parameters()};tick=time.time()
   with boundary.step(1):
    rh=s.state_digest(s.rng_capture(rng,'cuda'));o.zero_grad(set_to_none=True)
    for k,plan in enumerate(job['terminal_plan']['1']):
     d=folder/f'train_round1_seed{plan["seed"]}';ident=json.loads((d/'identity.json').read_text());assert ident['model_hash']==s.state_digest(state['model']) and ident['updates']==0
     g=np.load(d/'graph.npz');ed=g['edges'];ed=ed if ed.shape[0]==2 else ed.T
     eset={tuple(sorted(map(int,x))) for x in ed.T};bits=np.array([tuple(sorted(map(int,x))) in eset for x in query.T]);assert bits.sum()==len(eset)
     target=torch.as_tensor(bits,device='cuda').bool()
     for t in sorted(plan['times'],reverse=True):
      check();xp=d/f'Xt_{t:04d}.npz';sources[str(xp)]=s.base.sha(xp);zz=np.load(xp);assert int(zz['t'])==t
      ed,ff=w.common.state_edges(case,zz['xt'])
      logits,prob=w.runtime.predict(m,case['space'].types,ed,ff,query,fam,t/1000,details=True);calls+=1
      w.common.check_prob_device(prob,fam,case,m.prior)
      lp=torch.log_softmax(logits.double(),dim=-1)
      (.01*float(adv[k])*torch.where(target,lp[:,1],lp[:,0]).sum()/32).backward();backs+=1
    assert rh==s.state_digest(s.rng_capture(rng,'cuda'))
    assert all(p.grad is None for n,p in m.named_parameters() if not selected(n))
    norm=float(torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True))
    boundary.mark_optimizer_started();o.step();e.update(m,1);e.at_update_boundary=True
    assert all(torch.isfinite(p).all() for p in m.parameters())
   update_seconds=time.time()-tick
   assert all(torch.equal(p.detach().cpu(),before[n]) for n,p in m.named_parameters() if not selected(n))
   assert all(torch.equal(v.detach().cpu(),state['model'][n]) for n,v in m.named_buffers())
   full=torch.load(PREV/f'{job["id"]}_robust_temporary.pt',map_location='cpu',weights_only=False)
   da=db=dot=0.
   for n,p in m.named_parameters():
    a=p.detach().cpu().double()-before[n].double();b=full['model'][n].double()-before[n].double()
    da+=float(a.square().sum());db+=float(b.square().sum());dot+=float((a*b).sum())
   validation=[];boundary_hash=s.state_digest(boundary.checkpoint(binding))
   with torch.no_grad():
    for inp in fixed:
     check();logits,prob=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
     w.common.check_prob_device(prob,inp['f'],case,m.prior)
     terms=s.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[torch.as_tensor(inp['f'],device='cuda')]).cpu().numpy()
     pp=prob.cpu().double().numpy();q=inp['query'];f=inp['f'];lab=inp['labels'];n=len(case['space'].types)
     pred=np.zeros((n,case['n_families']));ref=np.zeros_like(pred)
     for endpoint in q:np.add.at(pred,(endpoint,f),pp);np.add.at(ref,(endpoint,f),lab)
     rel=[]
     for r in range(case['n_families']):
      pos=(f==r)&(lab==1);neg=(f==r)&(lab==0)
      rel.append(dict(family=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None,expected_edges=float(pp[f==r].sum()),reference_edges=int(pos.sum()),missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum()),node_relation_degree_MAE=float(abs(pred[:,r]-ref[:,r]).mean())))
     dg=abs(pred.sum(1)-ref.sum(1));types=[]
     for ty in sorted(set(case['space'].types.tolist())):
      ix=case['space'].types==ty;rd=ref.sum(1);hub=ix&(rd>=np.quantile(rd[ix],.9));low=ix&(rd<=1)
      types.append(dict(type=int(ty),low_n=int(low.sum()),hub_n=int(hub.sum()),low_MAE=float(dg[low].mean()) if low.any() else None,hub_MAE=float(dg[hub].mean()) if hub.any() else None))
     validation.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=float(terms.sum()/case['space'].total),node_total_degree_MAE=float(dg.mean()),relations=rel,type_groups=types))
   assert boundary_hash==s.state_digest(boundary.checkpoint(binding))
   saved=boundary.checkpoint(binding);cp=OUT/f'{job["id"]}_partial_temporary.pt';torch.save(saved,cp)
   reloaded=torch.load(cp,map_location='cpu',weights_only=False);s.restore_payload(reloaded,m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'))
   assert s.state_digest(boundary.checkpoint(binding))==s.state_digest(saved)
   result=dict(job=job['id'],dataset=ds,seed=seed,trainable_names=names,trainable_count=sum(p.numel() for p in params),total_parameter_count=sum(p.numel() for p in m.parameters()),gradient_norm_before_clip=norm,clip_scale=min(1,1/(norm+1e-6)),update_seconds=update_seconds,partial_update_norm=da**.5,full_update_norm=db**.5,update_cosine=dot/max((da*db)**.5,1e-300),frozen_raw_exact=True,buffers_exact=True,restore_exact=True,RNG_unchanged=True,EMA_updates=e.updates,validation=validation,baseline=old['baseline'],full_validation=old['arms']['robust']['validation'],evaluation_panel_already_seen=True)
   write(job['id']+'.json',result);done.append(job['id']);write('progress.json',dict(conditions=done,forwards=calls,backwards=backs))
   for path in [folder/'initial_complete.pt',Path(row['data']),Path(row['readout']),PREV/f'{job["id"]}_robust_temporary.pt',PREV/f'{job["id"]}.json']:sources[str(path)]=s.base.sha(path)
   del m,o,e,params,state,parent,initial,loaded,before,full,saved,reloaded;torch.cuda.empty_cache()
  write('completion.json',dict(status='complete',conditions=done,forwards=calls,backwards=backs,temporary_updates=4,new_graphs=0,end_epoch=time.time(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20))
 except BaseException:write('failure.json',dict(error=traceback.format_exc(),conditions=done,forwards=calls,backwards=backs,end_epoch=time.time()));raise
 finally:write('sources.json',sources)
def launch():
 OUT.mkdir(parents=True,exist_ok=True)
 write('protocol.json',dict(start_epoch=START,science_stop_epoch=STOP,delivery_epoch=START+900,max_GPU_hours=.25,source='GDPO partial parameter principle, not official replication',selection='backbone.tf_layers.[4-7].self_attn.* only',score='frozen robust_heat_v1',conditions=['IMDB0411','IMDB0412','PubMed0421','PubMed0422'],new_updates=4,backward_calls=128,evaluation_calls=24,old_full_parameter_updates_reused=4,old_evaluation_panel_reused=True,limits='single-step raw conditional proxies, no complete graphs or EMA quality; cannot isolate update magnitude from subspace',lr=1e-5,clip_norm=1,new_graphs=0))
 if time.time()>STOP-240:raise RuntimeError('insufficient time for all four conditions')
 used=sum(json.loads(f.read_text())['GPU_hours'] for f in OUT.glob('GPU_release_attempt*.json'))*3600
 start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with (OUT/'worker.log').open('w') as f:
  child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write('launch.json',dict(pid=child.pid,GPU=0,allocation_start=start,science_stop_epoch=STOP,delivery_epoch=START+900))
  try:child.wait(timeout=max(.1,min(STOP-start,900-used)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  finally:write('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,allocation_start=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600))
 if child.returncode:sys.exit(child.returncode)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:launch()
