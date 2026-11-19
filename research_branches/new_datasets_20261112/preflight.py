"""Bounded real GPU integration/cost checks; temporary models only."""
import os,sys,time,copy,traceback,signal
from pathlib import Path
from support import *
def main():
 out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);start=time.time();records=[]
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('preflight 8 minute limit')));signal.alarm(480)
 try:
  setup();base.write(out/'environment.json',w.common.environment('cuda'))
  for ds in ('DBLP_HGEN','ACM_HAN_full7'):
   seed=2026111211;case=case_for(ds);cfg=config(ds,seed);seed_all(seed)
   m=w.StructuralDenoiser(case,cfg,'A').cuda();o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'});e=s.ParameterEMA(m,.99)
   seed_all(seed+1);rng=np.random.default_rng(np.random.SeedSequence([2026111202,seed]));b=s.TrainingBoundary(m,o,e,rng,'cuda',0);binding=dict(dataset=ds,config=cfg,temporary=True)
   initial=b.checkpoint(binding);times=[]
   ids,fam,q=w.common.plan_for(case,2026111221,1).block(0);labels=np.isin(ids,case['clean_ids']).astype(np.uint8);score=Score(ds,case,q,fam,labels);assert abs(sum(score(labels)[:2]))<1e-12
   for pos in range(1,7):
    tick=time.time()
    with b.step(pos):
     inp=single.one_input(case,cfg,pos,rng);row=single.update(m,o,case,inp,1000,'cuda',e,pos-1,b)
    times.append(time.time()-tick)
   state=b.checkpoint(binding);torch.save(state,out/(ds+'_temporary.pt'));disk=torch.load(out/(ds+'_temporary.pt'),map_location='cpu',weights_only=False);assert s.state_digest(state)==s.state_digest(disk)
   want=single.one_input(case,cfg,7,copy.deepcopy(rng));s.restore_payload(disk,m,o,e,binding,lambda x:s.rng_restore(x,rng,'cuda'));b=s.TrainingBoundary(m,o,e,rng,'cuda',6)
   assert s.state_digest(b.checkpoint(binding))==s.state_digest(state)
   got=single.one_input(case,cfg,7,copy.deepcopy(rng));assert s.input_hash([got])==s.input_hash([want])
   before=s.state_digest(b.checkpoint(binding))
   def ev(_):
    with torch.no_grad():return w.runtime.predict(m,case['space'].types,got['edges'],got['edge_families'],got['query'],got['f'],got['t']/1000).detach().cpu()
   with b.validation():s.ema_validation(ev,m,e,boundary=b)
   assert before==s.state_digest(b.checkpoint(binding))
   with b.step(7):
    actual=single.one_input(case,cfg,7,rng);assert s.input_hash([actual])==s.input_hash([want]);single.update(m,o,case,actual,1000,'cuda',e,6,b)
   m.eval();gen=s.RecordedGenerator(2026111221,case['space'].total);allf=np.repeat(np.arange(case['n_families']),case['space'].counts);x=(gen.random(case['space'].total)<case['prior'][allf]).astype(np.uint8)
   kernel=w.MarginalKernel(case['prior'],w.common.schedule(1000));chain=w.SamplingBoundary(x,1000,gen,'cuda');rt=[]
   for t in range(1000,980,-1):
    tick=time.time();rec=chain.advance(lambda a,at:w.runtime.reverse_step(m,case,a,kernel,at,1,2026111221,gen,'cuda'),lambda a,r:None)
    assert rec['exact_coverage'] and rec['covered_slots']==len(ids);assert not chain.state[~np.repeat(case['active'],case['space'].counts)].any();rt.append(time.time()-tick)
   row=dict(dataset=ds,seed=seed,temporary_updates=7,temporary_reverse_steps=20,full_chains=0,types=len(np.unique(case['space'].types)),relations=case['n_families'],active_slots=len(ids),full_slots=case['space'].total,parameters=sum(p.numel() for p in m.parameters()),update_seconds=times,reverse_seconds=rt,loaded_state_exact=True,next_input_RNG_exact=True,EMA_restores_raw=True,recomputed_GPU_update_bitwise_not_certified=True,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,score_scaling=score.s)
   records.append(row);base.write(out/'conditions.json',records)
   del m,o,e,b,state,disk,initial;torch.cuda.empty_cache()
  base.write(out/'receipt.json',dict(status='passed',conditions=records,start_epoch=start,end_epoch=time.time(),seconds=time.time()-start))
 except BaseException:
  base.write(out/'receipt.json',dict(status='failed',conditions=records,error=traceback.format_exc(),start_epoch=start,end_epoch=time.time(),seconds=time.time()-start));raise
if __name__=='__main__':main()
