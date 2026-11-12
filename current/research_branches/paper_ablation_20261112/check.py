"""Bounded CPU contract and optional GPU update/restore checks. No quality gate."""
import os,sys,json,time,copy,traceback,signal
from pathlib import Path
from scoring import *
def main():
 out=Path(sys.argv[1]);gpu='--gpu' in sys.argv;folder=out/('gpu_check' if gpu else 'cpu_check');folder.mkdir(exist_ok=False)
 p=json.loads((out/'protocol.json').read_text());start=time.time();records=[]
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded preflight')));signal.alarm(480)
 def save(name,x):base.write(folder/name,x)
 try:
  for path,h in p['bindings'].items():assert base.sha(path)==h,path
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  if gpu:save('environment.json',w.common.environment('cuda'))
  for pid,row in p['parents'].items():
   if gpu and row['train_seed'] not in (2026110411,2026110421):continue
   case=case_from(row['data']);z=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=z['binding']['config']
   m=w.StructuralDenoiser(case,cfg,row['architecture']);m.load_state_dict(z['model'],strict=True);m.eval()
   assert torch.equal(m.prior.cpu(),torch.as_tensor(case['prior'],dtype=m.prior.dtype))
   ids,fam,q=w.common.plan_for(case,2026111402,1).block(0);labels=np.isin(ids,case['clean_ids']).astype(np.uint8)
   typed=TypedScore(row['dataset'],case,q,fam,labels);generic=GenericScore(row['dataset'],case,q,fam,labels)
   assert max(abs(sum(typed(labels)[:2])),abs(sum(generic(labels)[:2])))<1e-12
   oldgraph=next(x for x in p['reused_graphs'] if x['parent']==pid);g=np.load(oldgraph['path']);ee=g['edges'];ee=ee.T if ee.shape[0]!=2 else ee
   ff,ll=case['space'].encode(ee);b=np.zeros(case['space'].total,np.uint8);b[case['space'].offsets[ff]+ll]=1;bits=b[ids]
   scores=[typed(bits)[:2],generic(bits)[:2]];assert np.isfinite(scores).all() and np.min(scores)>-1e-10
   # Same type-preserving permutation of reference and candidate must not change either score.
   perm=np.arange(len(case['space'].types));rng=np.random.default_rng(2026111229)
   for t in np.unique(case['space'].types):
    ix=np.flatnonzero(case['space'].types==t);perm[ix]=rng.permutation(ix)
   typ2=TypedScore(row['dataset'],case,perm[q],fam,labels);gen2=GenericScore(row['dataset'],case,perm[q],fam,labels)
   assert np.max(abs(np.array(scores)-np.array([typ2(bits)[:2],gen2(bits)[:2]])))<1e-10
   rowout=dict(parent=pid,parameters=sum(x.numel() for x in m.parameters()),schema_prior_and_strict_load=True,
       score_self_zero=True,type_preserving_permutation=True,scores=scores)
   if gpu:
    m.cuda();optcfg={k:v for k,v in cfg['optimizer'].items() if k!='type'};optcfg['lr']=1e-5
    o=torch.optim.AdamW(m.parameters(),**optcfg);e=s.ParameterEMA(m,.99)
    rng=np.random.default_rng(2026111230);boundary=s.TrainingBoundary(m,o,e,rng,'cuda',0);binding=dict(parent=pid,temporary=True)
    def inp(pos,rng_):return w.common.train_input(case,1000,1,row['train_seed'],pos,rng_,force_t=500)
    def update(pos):
     tick=time.time();o.zero_grad(set_to_none=True)
     with boundary.step(pos):
      a=inp(pos,rng);logits,prob=w.runtime.predict(m,case['space'].types,a['edges'],a['edge_families'],a['query'],a['f'],a['t']/1000,details=True)
      w.common.check_prob_device(prob,a['f'],case,m.prior)
      # Exercise exact clean-head cost-advantage implementation on a saved graph.
      order=np.argsort(a['ids']);ref_order=np.argsort(ids);target=bits[ref_order][np.argsort(order)]
      assert np.array_equal(a['ids'][order],ids[ref_order])
      lp=torch.log_softmax(logits.double(),-1);target=torch.as_tensor(target,device='cuda').bool()
      loss=.01*sum(scores[0])*torch.where(target,lp[:,1],lp[:,0]).sum()/16
      loss.backward();assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
      grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));boundary.mark_optimizer_started();o.step();e.update(m,pos);e.at_update_boundary=True
     return dict(seconds=time.time()-tick,loss=float(loss.detach()),gradient_norm=grad,input_hash=s.input_hash([a]))
    u1=update(1);state=boundary.checkpoint(binding);torch.save(state,folder/(pid+'.pt'));loaded=torch.load(folder/(pid+'.pt'),map_location='cpu',weights_only=False)
    assert s.state_digest(state)==s.state_digest(loaded)
    next_input=s.input_hash([inp(2,copy.deepcopy(rng))]);s.restore_payload(loaded,m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'));boundary=s.TrainingBoundary(m,o,e,rng,'cuda',1)
    assert s.state_digest(boundary.checkpoint(binding))==s.state_digest(state)
    assert s.input_hash([inp(2,copy.deepcopy(rng))])==next_input
    u2=update(2);assert u2['input_hash']==next_input
    before=s.state_digest(boundary.checkpoint(binding))
    with boundary.validation():s.ema_validation(lambda _:sum(float(x.detach().sum()) for x in m.parameters()),m,e,boundary=boundary)
    assert before==s.state_digest(boundary.checkpoint(binding))
    gen=s.RecordedGenerator(p['sample_seeds'][0],case['space'].total);allf=np.repeat(np.arange(case['n_families']),case['space'].counts)
    x=(gen.random(case['space'].total)<case['prior'][allf]).astype(np.uint8);chain=w.SamplingBoundary(x,1000,gen,'cuda');kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000));ts=[]
    for _ in range(12):
     tick=time.time();rec=chain.advance(lambda a,t:w.runtime.reverse_step(m,case,a,kernel,t,1,p['sample_seeds'][0],gen,'cuda'),lambda a,r:None)
     assert rec['exact_coverage'] and rec['covered_slots']==len(ids);ts.append(time.time()-tick)
    rowout.update(temporary_updates=[u1,u2],reverse_step_seconds=ts,complete_state_reload_exact=True,next_input_RNG_exact=True,
        CUDA_update_recomputation_bitwise_not_certified=True,EMA_restores_raw=True,peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    del o,e,boundary,state,loaded;torch.cuda.empty_cache()
   records.append(rowout);save('conditions.json',records);del m,z
  save('receipt.json',dict(status='passed',GPU=gpu,start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,conditions=len(records)))
 except BaseException:
  save('receipt.json',dict(status='failed',GPU=gpu,start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,error=traceback.format_exc()));raise
if __name__=='__main__':main()
