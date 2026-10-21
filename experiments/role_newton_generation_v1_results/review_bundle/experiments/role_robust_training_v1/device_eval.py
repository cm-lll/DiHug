"""Device-portable exports of the SAME historical validation/response formulas."""
import random
import time
import numpy as np
import torch
from core import c,predict
from hsgd.losses import supported_binary_ce
from hsgd.computational import FrozenTypedContext

def validate_device(model,values,cfg,targets,check):
    device=next(model.parameters()).device.type;rng=c.rng_save(np.random.default_rng(0),device)
    training=model.training;model.eval();rows=[];c.sync(device);start=time.monotonic()
    try:
        with torch.no_grad():
            for identifier,condition,case,inp,index in values:
                check();model.set_target(targets[condition]);plan=c.plan_for(case,[cfg["validation_seed"],index,21],cfg["rounds"])
                ce=np.zeros((case["n_families"],2));counts=np.zeros_like(ce,dtype=int);expected=np.zeros(case["n_families"])
                for b in range(cfg["rounds"]):
                    check();ids,f,q=plan.block(b)
                    logits,p=predict(model,case["space"].types,inp["edges"],inp["edge_families"],q,f,inp["t"]/cfg["T"],details=True)
                    y=np.isin(ids,case["clean_ids"]).astype(np.float32)
                    item=supported_binary_ce(logits,torch.as_tensor(y,device=logits.device),model.prior[torch.as_tensor(f,device=logits.device)]).detach().cpu().numpy()
                    prob=c.check_prob(p,f,case);np.add.at(expected,f,prob)
                    for rel in range(case["n_families"]):
                        for label in (0,1):
                            ix=(f==rel)&(y==label);counts[rel,label]+=int(ix.sum());ce[rel,label]+=float(item[ix].sum())
                rows.append(dict(id=identifier,condition=condition,t=inp["t"],ce=float(ce.sum()/case["space"].total),
                    ce_sum_by_family_label=ce.tolist(),counts_by_family_label=counts.tolist(),expected_edges=expected.tolist(),
                    expected_edge_bias=(expected-case["counts"]).tolist(),normalization="full schema S; inactive zero slots have loss0"))
    finally:
        c.rng_restore(rng,np.random.default_rng(0),device);model.train(training)
    c.sync(device)
    return dict(seconds=time.monotonic()-start,rows=rows,mean_ce=float(np.mean([r["ce"] for r in rows])))

def response_device(models,values,targets,cfg,check):
    rows=[];arrays={}
    for arm,m in models.items():
        m.eval()
        for identifier,cond,case,inp,index in values[:3]:
            plan=c.plan_for(case,[cfg["validation_seed"],index,21],4);ps=[];ls=[]
            for target in ("balanced","extreme","balanced"):
                check();m.set_target(targets[target]);ll=[];pp=[];ff=[];ii=[]
                ctx=FrozenTypedContext(m,case["space"].types,inp["edges"],inp["edge_families"],"response")
                try:
                    with torch.no_grad():
                        for b in range(4):
                            ids,f,q=plan.block(b)
                            _,l,p=ctx.predict(q,f,inp["t"]/64,"response",details=True)
                            ii.append(ids);ll.append(l.detach().cpu().numpy());pp.append(p.detach().cpu().numpy());ff.append(f)
                finally:ctx.close()
                order=np.argsort(np.concatenate(ii));ls.append(np.concatenate(ll)[order]);ps.append(np.concatenate(pp)[order]);mg=np.concatenate(ff)[order]==8
            delta=ps[1]-ps[0];gap=(ls[1][:,1]-ls[1][:,0])-(ls[0][:,1]-ls[0][:,0])
            rows.append(dict(arm=arm,t=inp["t"],input_id=identifier,p0_max_abs=float(abs(delta).max()),
                MG_p0_max_abs=float(abs(delta[mg]).max()),MG_p0_mean_abs=float(abs(delta[mg]).mean()),
                MG_logit_gap_max_abs=float(abs(gap[mg]).max()),repeat_max_abs=float(abs(ps[2]-ps[0]).max())))
            for j,p in enumerate(ps):arrays[f"{arm}_t{inp['t']}_target{j}_p0"]=p
            for j,l in enumerate(ls):arrays[f"{arm}_t{inp['t']}_target{j}_logits"]=l
    return rows,arrays
