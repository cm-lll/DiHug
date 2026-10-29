"""Full-query sequential normal2: one zero_grad, two half backwards, one step."""
import time,numpy as np,torch
from hsgd.bw_preflight.common import train_input,check_prob_device,cpu_tree
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from count_aux import count_loss,case_contract,LAMBDA
def pair_inputs(case,cfg,position,rng,force_t=None):
    return [train_input(case,cfg['T'],1,cfg['seed'],2*position+j,rng,force_t=force_t) for j in range(2)]
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
def update_observation(before,after,step):
    """A watched tensor is a diagnostic, not an every-update success criterion."""
    changed=float((after.detach()-before).abs().max())
    if not np.isfinite(changed):raise FloatingPointError('nonfinite observed parameter change')
    return dict(changed=changed,observed_parameter='backbone.lin_out_E.weight',
                observed_parameter_changed=bool(changed>0) if step else None,
                update_note='watched_parameter_unchanged_not_failure' if step and changed==0 else None)
def normal2(model,opt,case,inputs,T,device,step=True,diagnostics=None,lambda_count=None,observe=False):
    if len(inputs)!=2:raise ValueError('exactly two inputs')
    sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True);model.train()
    weight=float(getattr(model,'count_aux_weight',0.) if lambda_count is None else lambda_count)
    if weight not in (0.,LAMBDA):raise ValueError('unfrozen auxiliary weight')
    from gradient_observation import GradientObservation
    observer=GradientObservation(model) if observe else None
    observation_seconds=0.
    total=0.;ce_total=0.;aux_total=0.;rows=[];expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
    for index,inp in enumerate(inputs):
        if not np.array_equal(np.sort(inp['ids']),expected) or inp['ce_weight']!=1/case['space'].total:
            raise ValueError('full query once / whole-schema denominator required')
        token=('normal2',index,id(inp));ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
        sync(device);begin=time.perf_counter()
        try:
            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,token,details=True)
            check_prob_device(p,inp['f'],case,model.prior)
            labels=torch.as_tensor(inp['labels'],device=device)
            ce=supported_binary_ce(logits,labels,model.prior[graph.query_families]).sum()/case['space'].total
            target,slots,fixed=case_contract(case,inp)
            aux,counts=count_loss(p,graph.query_families,target,slots,fixed)
            # Preserve the original CE arithmetic and gradient when lambda=0.
            loss=ce/2 if weight==0 else ce/2+(weight/2)*aux
            ce_total+=float(ce.detach())/2;aux_total+=float(aux.detach())/2
            count_record={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in counts.items()}
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite half loss')
            sync(device);forward=time.perf_counter()-begin
            if diagnostics is not None:
                diagnostics.setdefault('inputs',[]).append(dict(logits=cpu_tree(logits),p0=cpu_tree(p),loss=cpu_tree(loss)))
            if observer is not None:
                stamp=time.perf_counter();observer.add(ce/2,(LAMBDA/2)*aux)
                sync(device);observation_seconds+=time.perf_counter()-stamp
            begin=time.perf_counter();loss.backward();sync(device);backward=time.perf_counter()-begin
            total+=float(loss.detach());rows.append(dict(t=inp['t'],slots=len(inp['ids']),forward_seconds=forward,backward_seconds=backward,ce=float(ce.detach()),auxiliary=float(aux.detach()),weighted_auxiliary=weight*float(aux.detach()),counts=count_record))
        finally:ctx.close()
        del graph,logits,p,loss,labels,ctx,ce,aux,counts
    gradients={n:p.grad for n,p in model.named_parameters()}
    live=[v for v in gradients.values() if v is not None]
    if not live or not all(torch.isfinite(v).all() for v in live):raise FloatingPointError('nonfinite/no gradient')
    norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v) for v in live])))
    if diagnostics is not None:diagnostics.update(gradients=cpu_tree(gradients),loss=torch.tensor(total,dtype=torch.float64))
    before=model.backbone.lin_out_E.weight.detach().clone();begin=time.perf_counter()
    if step:opt.step()
    sync(device);optimizer_seconds=time.perf_counter()-begin
    observation=update_observation(before,model.backbone.lin_out_E.weight,step)
    return dict(loss=total,ce=ce_total,auxiliary=aux_total,weighted_auxiliary=weight*aux_total,lambda_count=weight,inputs=rows,gradient_norm=norm,optimizer_seconds=optimizer_seconds,
                **observation,normal2_seconds=time.perf_counter()-start,optimizer_steps=int(step),feature_calls=model.feature_calls,
                gradient_observation=observer.result() if observer is not None else None,observation_seconds=observation_seconds)
