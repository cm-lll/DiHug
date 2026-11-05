"""Original normal1 update plus one frozen role-distribution loss from same logits."""
import time,numpy as np,torch
from hsgd.bw_preflight.common import check_prob_device
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from normal2 import sync,update_observation
from .role_loss import LAMBDA

def update(model,opt,case,inp,T,device,ema,completed,transaction=None,*,objective,diagnostic=False):
    if completed!=ema.updates or not ema.at_update_boundary:raise ValueError('raw/EMA boundary')
    expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
    if not np.array_equal(np.sort(inp['ids']),expected) or inp['ce_weight']!=1/case['space'].total:raise ValueError('full R1 query/schema denominator')
    sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True);model.train()
    token=('single_input_output',completed,id(inp));ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
    try:
        graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,token,details=True)
        check_prob_device(p,inp['f'],case,model.prior)
        labels=torch.as_tensor(inp['labels'],device=device)
        ce=supported_binary_ce(logits,labels,model.prior[graph.query_families]).sum()/case['space'].total
        tick=time.perf_counter();aux,detail=objective(p,inp['query']);sync(device);aux_seconds=time.perf_counter()-tick
        loss=ce+LAMBDA*aux
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite total loss')
        diagnostics={}
        if diagnostic:
            ag=torch.autograd.grad(LAMBDA*aux,logits,retain_graph=True)[0]
            cg=torch.autograd.grad(ce,logits,retain_graph=True)[0]
            diagnostics=dict(aux_logit_gradient_norm=float(torch.linalg.vector_norm(ag)),ce_logit_gradient_norm=float(torch.linalg.vector_norm(cg)),aux_nonzero=int(torch.count_nonzero(ag)))
        sync(device);forward=time.perf_counter()-start;loss.backward();sync(device)
        live=[p.grad for p in model.parameters() if p.grad is not None]
        if not live or not all(torch.isfinite(x).all() for x in live):raise FloatingPointError('nonfinite/no gradients')
        norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in live])))
        before=model.backbone.lin_out_E.weight.detach().clone()
        if transaction is not None:transaction.mark_optimizer_started()
        ema.at_update_boundary=False;tick=time.perf_counter();opt.step();sync(device);opt_seconds=time.perf_counter()-tick
        observation=update_observation(before,model.backbone.lin_out_E.weight,True)
        tick=time.perf_counter();ema.update(model,completed+1);ema.at_update_boundary=True;sync(device)
        return dict(loss=float(loss.detach()),unweighted_CE=float(ce.detach()),auxiliary_loss=float(aux.detach()),weighted_auxiliary=float(LAMBDA*aux.detach()),auxiliary_seconds=aux_seconds,role_detail=detail,
          gradient_norm=norm,optimizer_seconds=opt_seconds,EMA_seconds=time.perf_counter()-tick,forward_seconds=forward,update_seconds=time.perf_counter()-start,
          optimizer_steps=1,ema_updates=ema.updates,inputs=1,slots=len(inp['ids']),S=case['space'].total,CE_coefficient=1/case['space'].total,**observation,**diagnostics)
    finally:ctx.close()
