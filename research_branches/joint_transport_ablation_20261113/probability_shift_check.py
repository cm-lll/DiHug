"""Read-only aligned probabilities and analytic coupling, no optimizer/sampling."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,traceback,importlib.util,subprocess,signal,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/probability_shift_check_20261113T104256'
PANEL=ROOT/'results/joint_transport_ablation_20261113T011928'
FULL=ROOT/'results/robust_heat_update_check_20261113T095140'
PART=ROOT/'results/partial_attention_check_20261113T101200'
START=datetime.fromisoformat('2026-11-13T10:42:56+08:00').timestamp();STOP=START+480
def write(name,x):
 OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2))
def worker():
 spec=importlib.util.spec_from_file_location('shift_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
 s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);w,np,torch=s.w,s.np,s.torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 calls=0;done=[];sources={}
 def check():
  if time.time()>=STOP:raise TimeoutError('science deadline')
 try:
  proto=json.loads((PANEL/'protocol.json').read_text())
  for path,h in proto['bindings'].items():assert s.base.sha(path)==h,path
  write('environment.json',w.common.environment('cuda'))
  for job in proto['jobs']:
   if job['architecture']!='A':continue
   check();ds=job['dataset'];row=proto['parents'][job['parent']];seed=row['train_seed'];folder=PANEL/job['id']/f'{ds}_T_eager'
   old=json.loads((PART/(job['id']+'.json')).read_text())
   paths=dict(baseline=folder/'initial_complete.pt',full=FULL/f'{job["id"]}_robust_temporary.pt',partial=PART/f'{job["id"]}_partial_temporary.pt')
   parent=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=parent['binding']['config']
   z=np.load(row['data']);ed=z['edges'];ed=ed if ed.shape[0]==2 else ed.T
   case=w.common.make_case(z['types'],ed,tuple(map(tuple,z['schema'])));kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000))
   fixed=[w.common.train_input(case,1000,1,seed,920000+j,np.random.default_rng(np.random.SeedSequence([202611130953,seed,j])),force_t=t) for j,t in enumerate([1,75,250,550,800,1000])]
   assert [s.input_hash([x]) for x in fixed]==[r['input_hash'] for r in old['baseline']]
   m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval();m.requires_grad_(False)
   data={};rows=[];rng=np.random.default_rng(202611131043)
   for arm,path in paths.items():
    check();sources[str(path)]=s.base.sha(path);state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['updates']==(0 if arm=='baseline' else 1)
    m.load_state_dict(state['model'],strict=True);s.rng_restore(state['rng'],rng,'cuda')
    before=s.state_digest(m.state_dict());rh=s.state_digest(s.rng_capture(rng,'cuda'))
    assert before==s.state_digest(state['model'])
    with torch.no_grad():
     for j,inp in enumerate(fixed):
      check();tick=time.time();logits,prob=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);calls+=1
      w.common.check_prob_device(prob,inp['f'],case,m.prior)
      pp=prob.cpu().double().numpy();ll=logits.cpu().double().numpy()
      assert np.isfinite(ll).all()
      edge_set={tuple(sorted(map(int,e))) for e in inp['edges'].T}
      xt=np.array([tuple(sorted(map(int,e))) in edge_set for e in inp['query'].T],dtype=np.uint8)
      q=kernel.reverse(pp,xt,inp['f'],inp['t']-1,inp['t'])
      assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
      if inp['t']==1:assert np.allclose(pp,q,atol=1e-12,rtol=1e-12)
      data[f'{arm}_p_{j}']=pp;data[f'{arm}_q_{j}']=q;data[f'{arm}_logit_{j}']=ll[:,1]-ll[:,0]
      if arm=='baseline':
       data[f'family_{j}']=inp['f'];data[f'query_{j}']=inp['query'];data[f'label_{j}']=inp['labels'];data[f'xt_{j}']=xt;data[f'ids_{j}']=inp['ids'];data[f'types_{j}']=case['space'].types;data[f'prior_{j}']=case['prior']
      # Reproduce existing scalar readout without using it to select conditions.
      ref=np.zeros(len(case['space'].types));pred=np.zeros_like(ref)
      for endpoint in inp['query']:
       np.add.at(ref,endpoint,inp['labels']);np.add.at(pred,endpoint,pp)
      actual=float(abs(pred-ref).mean());oldkey={'baseline':'baseline','full':'full_validation','partial':'validation'}[arm]
      expected=old[oldkey][j]['node_total_degree_MAE']
      rows.append(dict(arm=arm,t=inp['t'],input_hash=s.input_hash([inp]),node_degree_MAE=actual,previous_MAE=expected,recompute_difference=actual-expected,seconds=time.time()-tick))
    assert before==s.state_digest(m.state_dict()) and rh==s.state_digest(s.rng_capture(rng,'cuda'))
    assert all(p.grad is None for p in m.parameters())
   np.savez_compressed(OUT/f'{job["id"]}_probabilities.npz',**data)
   write(job['id']+'.json',dict(dataset=ds,seed=seed,job=job['id'],rows=rows,model_unchanged=True,RNG_unchanged=True,gradients_absent=True,updates=0,new_graphs=0,states={a:str(p) for a,p in paths.items()}));done.append(job['id'])
   write('progress.json',dict(conditions=done,forwards=calls));del m,parent,state,data;torch.cuda.empty_cache()
  write('completion.json',dict(status='complete',conditions=done,forwards=calls,backwards=0,updates=0,new_graphs=0,end_epoch=time.time(),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20))
 except BaseException:write('failure.json',dict(error=traceback.format_exc(),conditions=done,forwards=calls,end_epoch=time.time()));raise
 finally:write('sources.json',sources)
def launch():
 OUT.mkdir(parents=True,exist_ok=True)
 write('protocol.json',dict(start_epoch=START,science_stop_epoch=STOP,delivery_epoch=START+600,max_GPU_hours=.1,forwards=72,conditions=['IMDB0411','IMDB0412','PubMed0421','PubMed0422'],arms=['baseline','full_robust','partial_robust'],times=[1,75,250,550,800,1000],same_previous_input_root=202611130953,evaluation_panel_already_seen=True,readout='raw',new_graphs=0,updates=0,backwards=0,analysis='per-slot probability changes, family/label groups, redistribution and shrink-fit; analytic shared-uniform clean/posterior coupling; no hard sampling'))
 if time.time()>STOP-150:raise RuntimeError('not enough time for complete panel')
 start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with (OUT/'worker.log').open('w') as f:
  child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write('launch.json',dict(pid=child.pid,GPU=0,allocation_start=start,science_stop_epoch=STOP,delivery_epoch=START+600))
  try:child.wait(timeout=max(.1,min(STOP-start,360)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  finally:write('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,allocation_start=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600))
 if child.returncode:sys.exit(child.returncode)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:launch()
