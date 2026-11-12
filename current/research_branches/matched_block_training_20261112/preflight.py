"""Temporary full-sweep backward and complete-state recovery; no quality gate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,traceback,copy
from pathlib import Path
import sweep
s,w,np,torch=sweep.s,sweep.w,sweep.np,sweep.torch
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(sys.argv[1]);p=json.loads((out/'protocol.json').read_text());rows=[];start=time.time();receipt={'status':'running'}
 folder=out/'preflight';folder.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 def check():
  if time.time()-start>600:raise TimeoutError('10min preflight ceiling')
 try:
  for path,h in p['bindings'].items():assert s.base.sha(path)==h,path
  for ds in ('IMDB','PubMed'):
   row=p['parents'][ds];z=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=z['binding']['config']
   data=np.load(row['data']);ee=data['edges'];ee=ee.T if ee.shape[0]!=2 else ee
   case=w.common.make_case(data['types'],ee,tuple(map(tuple,data['schema'])))
   m=w.StructuralDenoiser(case,cfg,'A').cuda();m.eval();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=s.ParameterEMA(m,.99)
   loaded=torch.load(Path(p['resume_parent'])/(ds+'_T_eager/final_complete.pt'),map_location='cpu',weights_only=False)
   saved={k:v for k,v in loaded.items() if k!='pilot_progress'};binding=saved['binding'];rng=np.random.default_rng()
   def reset():
    s.restore_payload(s.cpu_copy(saved),m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'));o.zero_grad(set_to_none=True);m.eval()
    b=s.TrainingBoundary(m,o,e,rng,'cuda',2);assert s.state_digest(b.checkpoint(binding))==s.state_digest(saved);return b
   for r in (10,20):
    b=reset();statepath=ROOT/f'results/block_generation_panel_20261112T015400/chains/{ds}_R{r}_S2026111341'
    zz=np.load(statepath/'Xt_0500.npz');xt=zz['xt'];edges,families=w.common.state_edges(case,xt)
    g=np.load(statepath/'graph.npz');ff,ll=case['space'].encode(g['edges']);target=np.zeros(case['space'].total,dtype=np.uint8);target[case['space'].offsets[ff]+ll]=1
    steps=[json.loads(x) for x in (statepath/'steps.jsonl').read_text().splitlines()];qh=steps[500]['query_hash'];assert steps[500]['time']==500
    tick=time.time();before=s.state_digest(s.rng_capture(rng,'cuda'))
    with b.step(3):
     result=sweep.feedback(m,case,edges,families,500,r,2026111341,target,.01/16,qh,check)
     assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
     norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));b.mark_optimizer_started();o.step();e.update(m,3);e.at_update_boundary=True
    torch.cuda.synchronize();seconds=time.time()-tick
    assert before==s.state_digest(s.rng_capture(rng,'cuda')) and b.completed==e.updates==3
    post=b.checkpoint(binding);tmp=folder/f'{ds}_R{r}_temporary_complete.pt';torch.save(post,tmp)
    disk=torch.load(tmp,map_location='cpu',weights_only=False);assert s.state_digest(post)==s.state_digest(disk)
    s.restore_payload(disk,m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'));b=s.TrainingBoundary(m,o,e,rng,'cuda',3)
    assert s.state_digest(b.checkpoint(binding))==s.state_digest(post)
    with b.validation():
     s.ema_validation(lambda _:sweep.predict_all(m,case,edges,families,500,r,2026111341,check)[1].sum().item(),m,e,boundary=b)
    assert s.state_digest(b.checkpoint(binding))==s.state_digest(post)
    b=reset();original=s.state_digest(b.checkpoint(binding))
    try:
     with b.step(3):
      for ids,fam,logits,prob,plan in sweep.blocks(m,case,edges,families,500,r,2026111341,check):
       logits.sum().backward();raise InterruptedError('injected pre-optimizer block failure')
    except InterruptedError:pass
    assert s.state_digest(b.checkpoint(binding))==original
    o.zero_grad(set_to_none=True);gen=s.RecordedGenerator(2026111341,case['space'].total);kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000));chain=w.SamplingBoundary(xt,500,gen,'cuda')
    rec=chain.advance(lambda x,t:w.runtime.reverse_step(m,case,x,kernel,t,r,2026111341,gen,'cuda',check_time=check),lambda candidate,record:None)
    assert rec['blocks']==r and rec['exact_coverage'] and chain.next_t==499
    rows.append(dict(dataset=ds,rounds=r,status='passed',full_backward=result,temporary_update_seconds=seconds,gradient_norm=norm,complete_restore_exact=True,pre_optimizer_failure_rollback_exact=True,EMA_validation_restored=True,sampling_step_seconds=rec['step_seconds'],GPU_recomputation_bitwise_not_certified=True))
    s.base.write(folder/'conditions.json',rows);b=reset();tmp.unlink()
   del m,o,e,z,loaded,saved;torch.cuda.empty_cache()
  receipt['status']='passed'
 except BaseException:receipt.update(status='failed',traceback=traceback.format_exc());raise
 finally:
  receipt.update(start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,conditions=rows,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,device=os.environ.get('CUDA_VISIBLE_DEVICES'),temporary_models_discarded=True)
  s.base.write(out/'preflight_receipt.json',receipt)
if __name__=='__main__':main()
