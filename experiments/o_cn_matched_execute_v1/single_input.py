"""The frozen full-query CE update, with one input and no normal2 half factor."""
import time,numpy as np,torch
from hsgd.bw_preflight.common import train_input,check_prob_device,cpu_tree
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from normal2 import sync,update_observation
def one_input(case,cfg,position,rng,force_t=None):
 return train_input(case,cfg['T'],1,cfg['seed'],position,rng,force_t=force_t)
def update(model,opt,case,inp,T,device,ema,completed):
 if completed!=ema.updates or not ema.at_update_boundary:raise ValueError('raw/EMA boundary')
 expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
 if not np.array_equal(np.sort(inp['ids']),expected) or inp['ce_weight']!=1/case['space'].total:raise ValueError('full query / schema denominator')
 sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True);model.train()
 token=('single_input',completed,id(inp));ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
 try:
  graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,token,details=True)
  check_prob_device(p,inp['f'],case,model.prior)
  labels=torch.as_tensor(inp['labels'],device=device)
  loss=supported_binary_ce(logits,labels,model.prior[graph.query_families]).sum()/case['space'].total
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite CE')
  sync(device);forward=time.perf_counter()-start;loss.backward();sync(device)
  live=[p.grad for p in model.parameters() if p.grad is not None]
  if not live or not all(torch.isfinite(x).all() for x in live):raise FloatingPointError('nonfinite/no gradients')
  norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in live])))
  before=model.backbone.lin_out_E.weight.detach().clone();ema.at_update_boundary=False
  tick=time.perf_counter();opt.step();sync(device);opt_seconds=time.perf_counter()-tick
  observation=update_observation(before,model.backbone.lin_out_E.weight,True)
  tick=time.perf_counter();ema.update(model,completed+1);ema.at_update_boundary=True;sync(device)
  return dict(loss=float(loss.detach()),gradient_norm=norm,optimizer_seconds=opt_seconds,
     EMA_seconds=time.perf_counter()-tick,forward_seconds=forward,update_seconds=time.perf_counter()-start,
     optimizer_steps=1,ema_updates=ema.updates,inputs=1,slots=len(inp['ids']),S=case['space'].total,
     CE_coefficient=1/case['space'].total,**observation)
 finally:ctx.close()
