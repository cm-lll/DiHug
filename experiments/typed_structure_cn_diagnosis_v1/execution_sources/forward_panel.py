"""At most eight fixed-weight full queries; no backward, optimizer, or sampling."""
import pathlib,json,sys,os,time,hashlib,traceback,resource,subprocess,signal
import numpy as np
B=pathlib.Path(__file__).resolve().parent;ROOT=B.parents[1]
SRC=ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_final_screen_v1'
RUN=ROOT/'results/typed_structure_final_screen_formal_v1_20261029T0325/run'
sys.path[:0]=[str(SRC),str(SRC/'package')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2))
panel=read(B/'frozen_panel.json');ledger=read(B/'ledger.json');out=B/'forward';out.mkdir(exist_ok=False)
started=time.time();attempted=0;completed=[];failure=None
def check():
 if time.time()>=ledger['deadline_epoch']-180:raise TimeoutError('three-minute delivery reserve')
 if (B/'STOP').exists():raise InterruptedError('STOP')
def alarm(sig,frame):raise TimeoutError('bounded fixed panel process limit')
signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,min(300,int(ledger['deadline_epoch']-time.time()-180))))
try:
 check();uuid=os.environ['CUDA_VISIBLE_DEVICES'];assert uuid=='GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7'
 apps=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True,timeout=20)
 assert not apps.strip(),'GPU occupied before diagnostic: '+apps
 resource_record=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=20)
 write(out/'resource_precheck.json',dict(uuid=uuid,owner_authorization='current user explicitly permits one available local GPU; previous local allocation retained',compute_apps=apps,device=resource_record,epoch=time.time()))
 import torch
 from model_io import configure
 from runtime_contract import source_binding,dictionary_expected
 from experiment_contract import read_dictionary
 from prepare import load_case
 from model_variants import StructuralDenoiser
 from hsgd.computational import FrozenTypedContext
 from hsgd.bw_preflight.common import schedule,check_prob
 from hsgd.kernel import MarginalKernel
 from hsgd.losses import supported_binary_ce
 env=configure();write(out/'environment.json',env)
 oldenv=read(RUN/'PubMed_s2026102912_C/environment.json')
 for k in ['python_version','torch_runtime_version','packages','device_name']:assert env[k]==oldenv[k],k
 assert source_binding()[1]==panel['source_sha256']
 case,cfg=load_case('PubMed');basis,basis_receipt=read_dictionary(SRC/'dictionaries','PubMed',dictionary_expected('PubMed'));kernel=MarginalKernel(case['prior'],schedule(1000))
 for arm in ['C','E']:
  check();path=pathlib.Path(panel['models'][arm]['path']);assert sha(path)==panel['models'][arm]['sha256']
  payload=torch.load(path,map_location='cpu');assert payload['readout']=='ema_0.99' and payload['updates']==3000 and payload['decay']==.99
  assert payload['binding']['source_sha256']==panel['source_sha256'] and payload['binding']['arm']==arm and payload['binding']['dataset']=='PubMed'
  model=StructuralDenoiser(case,cfg,arm,basis).cuda();model.load_state_dict(payload['model'],strict=True)
  assert all(torch.equal(v.cpu(),payload['model'][k]) for k,v in model.state_dict().items())
  model.eval();del payload
  historical=read(RUN/f'PubMed_s2026102912_{arm}/validation.json')[-1]['ema_0.99']['rows']
  for inp in panel['inputs']:
   check();p=pathlib.Path(inp['path']);assert sha(p)==inp['sha256'];x=dict(np.load(p,allow_pickle=False));t=int(x['t']);ids=x['ids'];f=x['f'];state=x['state']
   token=('fixed_diagnostic',arm,inp['name']);ctx=FrozenTypedContext(model,x['types'],x['edges'],x['edge_families'],token)
   assert attempted<8;attempted+=1;write(out/'attempts.json',dict(attempted_full_queries=attempted,completed_full_queries=len(completed),optimizer_steps=0,new_graphs=0))
   torch.cuda.synchronize();begin=time.perf_counter()
   try:
    with torch.no_grad():
     graph,logits,prob=ctx.predict(x['query'],f,t/1000,token,details=True)
     p0=check_prob(prob,f,case);L=logits.cpu().numpy();assert np.isfinite(L).all()
     ce=None
     if 'labels' in x:ce=supported_binary_ce(logits,torch.as_tensor(x['labels'],device='cuda'),model.prior[graph.query_families]).cpu().numpy()
    torch.cuda.synchronize();network_seconds=time.perf_counter()-begin
   finally:ctx.close()
   q=kernel.reverse(p0,state[ids],f,t-1,t);assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
   arr=dict(ids=np.arange(18336),state=state,t=np.int64(t),types=x['types'],schema=x['schema'],query=np.empty((2,18336),dtype=np.int64),family=np.empty(18336,dtype=np.int64),p0=np.empty(18336,dtype=np.float64),q=np.empty(18336,dtype=np.float64),logits=np.empty((18336,2),dtype=np.float32))
   for key,value in [('family',f),('p0',p0),('q',q),('logits',L)]:arr[key][ids]=value
   arr['query'][:,ids]=x['query']
   hist=None
   if ce is not None:
    arr['clean_label']=np.empty(18336,dtype=np.float32);arr['clean_label'][ids]=x['labels'];arr['ce']=np.empty(18336,dtype=np.float32);arr['ce'][ids]=ce
    old=next(r for r in historical if r['t']==t);prob32=p0.astype(np.float32);current=[float(prob32[f==i].sum()) for i in range(10)];hist=dict(total_ce=float(ce.sum()/18336),historical_total_ce=old['ce_sum_over_S'],current_p0=current,historical_p0=[r['expected_edges'] for r in old['family']],scope='aggregate comparison only; old slot probabilities not saved')
    atol=panel['historical_validation_comparison']['atol'];rtol=panel['historical_validation_comparison']['rtol']
    hist['p0_aggregate_pass']=bool(np.allclose(current,hist['historical_p0'],atol=atol,rtol=rtol));hist['ce_aggregate_pass']=bool(np.isclose(hist['total_ce'],hist['historical_total_ce'],atol=atol,rtol=rtol))
    assert hist['p0_aggregate_pass'] and hist['ce_aggregate_pass'],'historical aggregate mismatch'
   name=arm+'_'+inp['name'];file=out/(name+'.npz');np.savez_compressed(file,**arr)
   rec=dict(arm=arm,input=inp['name'],t=t,input_sha256=inp['sha256'],weight_sha256=panel['models'][arm]['sha256'],output=str(file),output_sha256=sha(file),network_full_query_seconds=network_seconds,slots=18336,one_full_query=True,current_edges=int(state.sum()),p0_sum=float(p0.sum()),q_sum=float(q.sum()),birth=float(q[state[ids]==0].sum()),death=float((1-q[state[ids]==1]).sum()),historical_comparison=hist)
   rec['net']=rec['birth']-rec['death'];completed.append(rec);write(out/(name+'.json'),rec);write(out/'progress.json',dict(attempted=attempted,completed=len(completed),rows=completed));print(json.dumps({k:rec[k] for k in ['arm','input','p0_sum','birth','death','net']}),flush=True)
  del model;torch.cuda.empty_cache()
except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc())
finally:
 signal.alarm(0)
 result=dict(status='complete' if len(completed)==8 and failure is None else 'incomplete',attempted_full_queries=attempted,completed_full_queries=len(completed),optimizer_steps=0,backwards=0,sampled_graphs=0,started_epoch=started,finished_epoch=time.time(),rows=completed,failure=failure,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 if 'torch' in globals() and torch.cuda.is_initialized():result.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 write(out/'receipt.json',result)
if failure:raise RuntimeError(failure['error'])
