"""Full-query sequential normal2: one zero_grad, two half backwards, one step."""
import time,numpy as np,torch
from hsgd.bw_preflight.common import train_input,check_prob_device,cpu_tree
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
def pair_inputs(case,cfg,position,rng,force_t=None):
    return [train_input(case,cfg['T'],1,cfg['seed'],2*position+j,rng,force_t=force_t) for j in range(2)]
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
def normal2(model,opt,case,inputs,T,device,step=True,diagnostics=None):
    if len(inputs)!=2:raise ValueError('exactly two inputs')
    sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True);model.train()
    total=0.;rows=[];expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
    for index,inp in enumerate(inputs):
        if not np.array_equal(np.sort(inp['ids']),expected) or inp['ce_weight']!=1/case['space'].total:
            raise ValueError('full query once / whole-schema denominator required')
        token=('normal2',index,id(inp));ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
        sync(device);begin=time.perf_counter()
        try:
            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,token,details=True)
            check_prob_device(p,inp['f'],case,model.prior)
            labels=torch.as_tensor(inp['labels'],device=device)
            loss=supported_binary_ce(logits,labels,model.prior[graph.query_families]).sum()/(2*case['space'].total)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite half loss')
            sync(device);forward=time.perf_counter()-begin
            if diagnostics is not None:
                diagnostics.setdefault('inputs',[]).append(dict(logits=cpu_tree(logits),p0=cpu_tree(p),loss=cpu_tree(loss)))
            begin=time.perf_counter();loss.backward();sync(device);backward=time.perf_counter()-begin
            total+=float(loss.detach());rows.append(dict(t=inp['t'],slots=len(inp['ids']),forward_seconds=forward,backward_seconds=backward))
        finally:ctx.close()
        del graph,logits,p,loss,labels,ctx
    gradients={n:p.grad for n,p in model.named_parameters()}
    live=[v for v in gradients.values() if v is not None]
    if not live or not all(torch.isfinite(v).all() for v in live):raise FloatingPointError('nonfinite/no gradient')
    norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v) for v in live])))
    if diagnostics is not None:diagnostics.update(gradients=cpu_tree(gradients),loss=torch.tensor(total,dtype=torch.float64))
    before=model.backbone.lin_out_E.weight.detach().clone();begin=time.perf_counter()
    if step:opt.step()
    sync(device);optimizer_seconds=time.perf_counter()-begin
    changed=float((model.backbone.lin_out_E.weight-before).abs().max())
    if step and not changed>0:raise RuntimeError('expected parameter did not change')
    return dict(loss=total,inputs=rows,gradient_norm=norm,optimizer_seconds=optimizer_seconds,
                changed=changed,normal2_seconds=time.perf_counter()-start,optimizer_steps=int(step),feature_calls=model.feature_calls)
