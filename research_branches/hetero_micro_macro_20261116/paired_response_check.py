"""Frozen 96 read-only forwards; no updates or reverse draws."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,importlib.util,subprocess,signal,traceback,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
OUT=ROOT/'results/paired_response_check_20261116'
START=1794799501.0 # 2026-11-16 11:25:01 Asia/Shanghai
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def worker():
 spec=importlib.util.spec_from_file_location('support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
 s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);w,np,torch=s.w,s.np,s.torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 p=json.loads((SRC/'protocol.json').read_text());inventory=json.loads((SRC/'inventory.json').read_text());manifest=[];contexts=[];calls=0;bindings={}
 def bind(path):bindings[str(path)]=sha(path)
 def check():
  if time.time()>START+570:raise TimeoutError('wall stop')
 try:
  for path,h in p['runtime_source_bindings'].items():assert sha(path)==h,path
  bindings.update(p['runtime_source_bindings']);bind(SRC/'protocol.json');bind(SRC/'inventory.json');bind(__file__)
  # Freeze every input and query before looking at any new model output.
  for row in p['parents']:
   check();bind(row['data']);z=np.load(row['data']);e=z['edges'];e=e if e.shape[0]==2 else e.T
   case=w.common.make_case(z['types'],e,tuple(map(tuple,z['schema'])));space=case['space']
   fullq=np.concatenate([space.decode(f,np.arange(n)) for f,n in enumerate(space.counts)],axis=1)
   fullf=np.repeat(np.arange(case['n_families']),space.counts);inputs=[]
   for it in [i for i in inventory if i['model']==row['id'] and i['arm']=='parent']:
    directory=Path(it['directory']);files=list(directory.glob('Xt_*.npz'));assert len(files)==8
    path=min(files,key=lambda x:int(x.stem.split('_')[-1]));bind(path);zz=np.load(path);t=int(zz['t']);x=zz['xt'].astype(np.uint8)
    ident=json.loads((directory/'identity.json').read_text());assert ident['model_hash']==row['parent_model_hash'];bind(directory/'identity.json');bind(directory/'steps.jsonl')
    plan=s.RecordedPlan(w.common.plan_for(case,[it['sample_seed'],t,61],1));ids,fam,query=plan.block(0)
    old=next(json.loads(l) for l in (directory/'steps.jsonl').read_text().splitlines() if json.loads(l)['time']==t)
    assert plan.h.hexdigest()==old['query_hash']
    seed=[202611161701,row['train_seed'],it['sample_seed'],t]
    ne,nf,_=w.common.global_noisy_edges(space,case['clean'],case['prior'],w.common.schedule(1000)[t],np.random.default_rng(seed))
    ff,ll=space.encode(ne);normal=np.zeros(space.total,np.uint8);normal[space.offsets[ff]+ll]=1
    for kind,xx in [('free',x),('normal',normal)]:
     ei,ef=w.common.state_edges(case,xx);key=f'{row["id"]}_{it["sample_seed"]}_{kind}'
     dst=OUT/(key+'_input.npz');np.savez_compressed(dst,xt=xx,ids=ids,f=fam,query=query,fullquery=fullq,fullfamily=fullf,types=space.types,reference_edges=e,prior=case['prior'],t=t)
     entry=dict(key=key,model=row['id'],dataset=row['dataset'],train_seed=row['train_seed'],sample_seed=it['sample_seed'],kind=kind,t=t,source=str(path),input=str(dst),input_sha256=sha(dst),noise_seed=seed,query_hash=plan.h.hexdigest())
     manifest.append(entry);inputs.append((entry,xx,ei,ef,ids,fam,query))
   assert len(inputs)==8;contexts.append((row,case,inputs))
  assert len(manifest)==32;dump('input_manifest.json',manifest);dump('sources.json',bindings)
  dump('environment.json',w.common.environment('cuda'))
  for row,case,inputs in contexts:
   check();m=w.StructuralDenoiser(case,row['config'],'A').cuda().eval();m.requires_grad_(False)
   kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000));rng=np.random.default_rng(202611161702)
   for arm in ['parent','CE','calibrated_micro_macro']:
    check()
    if arm=='parent':path=row['initial_complete'];expected=row['parent_model_hash']
    else:
     cp=next(c for c in p['checkpoints'] if c['model']==row['id'] and c['arm']==arm and c['readout']=='ema');path=cp['checkpoint'];expected=cp['model_hash'];assert sha(path)==cp['checkpoint_sha256']
    bind(path);state=torch.load(path,map_location='cpu',weights_only=False);effective=dict(state['model'])
    if arm!='parent':
     assert state['updates']==state['ema']['updates']==128;effective.update(state['ema']['shadow'])
    assert s.state_digest(effective)==expected;m.load_state_dict(effective,strict=True);m.eval()
    before=s.state_digest(m.state_dict());rh=s.state_digest(s.rng_capture(rng,'cuda'))
    with torch.no_grad():
     for entry,x,ei,ef,ids,fam,query in inputs:
      check();logits,prob=w.runtime.predict(m,case['space'].types,ei,ef,query,fam,entry['t']/1000,details=True);calls+=1
      w.common.check_prob_device(prob,fam,case,m.prior);pp=prob.cpu().double().numpy();ll=logits.cpu().double().numpy();assert np.isfinite(ll).all()
      q=kernel.reverse(pp,x[ids],fam,entry['t']-1,entry['t']);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
      np.savez_compressed(OUT/(entry['key']+'_'+arm+'_prediction.npz'),p=pp,q=q,logits=ll)
      dump('progress.json',dict(forwards=calls,model=row['id'],arm=arm,epoch=time.time()))
    assert before==s.state_digest(m.state_dict()) and rh==s.state_digest(s.rng_capture(rng,'cuda'))
    assert all(v.grad is None for v in m.parameters());assert sha(path)==bindings[str(path)]
   del m,state,effective;torch.cuda.empty_cache()
  assert calls==96
  dump('completion.json',dict(forwards=calls,status='complete',updates=0,new_graphs=0,backwards=0,model_RNG_unchanged=True,peak_MiB=torch.cuda.max_memory_allocated()/2**20,epoch=time.time()))
 except BaseException:
  dump('failure.json',dict(forwards=calls,traceback=traceback.format_exc(),epoch=time.time()));raise
 finally:dump('sources.json',bindings)
def launch():
 OUT.mkdir(exist_ok=False)
 dump('protocol.json',dict(start_epoch=START,delivery_epoch=START+600,GPU_seconds_max=360,forwards=96,selection='lowest saved t in each of 16 fixed parent chains, independent of scores',normal_seed_root=202611161701,arms=['parentEMA3000','CE_EMA128','calibrated_micro_macro_EMA128'],off_policy=True,no_training=True,no_new_chain=True,script_sha256=sha(__file__)))
 if time.time()>START+300:raise RuntimeError('insufficient remaining wall time')
 start=time.time();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
 with (OUT/'worker.log').open('w') as f:
  child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump('launch.json',dict(pid=child.pid,GPU=0,allocation_start=start,hard_stop=min(start+360,START+570)))
  try:child.wait(timeout=max(.1,min(355,START+565-start)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  finally:dump('GPU_release.json',dict(exit_code=child.returncode,owned_process_exited=child.poll() is not None,start_epoch=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600))
 if child.returncode:sys.exit(child.returncode)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:launch()
