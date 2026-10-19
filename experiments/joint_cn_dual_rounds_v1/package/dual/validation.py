"""Frozen full-query same-reference denoising; independent RNG, no sampling."""
import time
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.runtime import predict
from hsgd.losses import supported_binary_ce
from .data import degrees

def panel(case,cfg):
    result=[]
    for j,t in enumerate(cfg['validation_times']):
        rng=np.random.default_rng([cfg['validation_seed'],j,99])
        inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['validation_seed'],j,rng,force_t=t)
        result.append(inp)
    return result

def run(model,case,cfg,device,inputs,check=lambda:None):
    saved=c.rng_save(np.random.default_rng(0),device); was_training=model.training
    started=time.monotonic();result=[];model.eval()
    try:
        with torch.no_grad():
            for j,inp in enumerate(inputs):
                n,r=len(case['space'].types),case['n_families']
                size=np.zeros((r,2),int);ce=np.zeros((r,2));prior_ce=np.zeros((r,2))
                expected=np.zeros(r);expected_deg=np.zeros((n,r));brier=0.;bins=np.zeros((10,3))
                # Exact-zero families need no network call but count in the full-schema panel.
                size[~case['active'],0]=case['space'].counts[~case['active']]
                bins[0,0]=int(case['space'].counts[~case['active']].sum())
                # Inactive families contribute exactly zero CE, but remain in full S denominator.
                plan=c.plan_for(case,[cfg['validation_seed'],j,21],cfg['rounds'])
                coverage=np.zeros(case['space'].total,np.uint8)
                for k in range(cfg['rounds']):
                    check();ids,f,q=plan.block(k)
                    if not len(ids):continue
                    logits,p=predict(model,case['space'].types,inp['edges'],inp['edge_families'],q,f,
                                     inp['t']/cfg['T'],backend=cfg['backend'],details=True)
                    y=np.isin(ids,case['clean_ids']).astype(np.float32)
                    pi=case['prior'][f];labels=torch.as_tensor(y,device=device)
                    item=supported_binary_ce(logits,labels,torch.as_tensor(pi,device=device)).cpu().numpy()
                    # Fixed families need finite dummy logits: exact support CE is zero.
                    lp=np.zeros((len(pi),2));free=(pi>0)&(pi<1)
                    lp[free,0]=np.log1p(-pi[free]);lp[free,1]=np.log(pi[free])
                    prior_logits=torch.as_tensor(lp,device=device)
                    base=supported_binary_ce(prior_logits,labels,torch.as_tensor(pi,device=device)).cpu().numpy()
                    prob=c.check_prob(p,f,case);coverage[ids]+=1
                    for rel in range(r):
                        for label in (0,1):
                            mask=(f==rel)&(y==label);size[rel,label]+=int(mask.sum())
                            ce[rel,label]+=float(item[mask].sum());prior_ce[rel,label]+=float(base[mask].sum())
                    np.add.at(expected,f,prob);np.add.at(expected_deg,(q[0],f),prob);np.add.at(expected_deg,(q[1],f),prob)
                    brier+=float(((prob-y)**2).sum())
                    bi=np.minimum((prob*10).astype(int),9)
                    np.add.at(bins[:,0],bi,1);np.add.at(bins[:,1],bi,prob);np.add.at(bins[:,2],bi,y)
                want=np.repeat(case['active'].astype(np.uint8),case['space'].counts)
                if not np.array_equal(coverage,want):raise ValueError('validation coverage mismatch')
                # Fixed-one families (if any) are active and are included by the plan.
                desc={}
                ref=degrees(case)
                for typ in range(case['n_types']):
                    ix=case['space'].types==typ
                    desc[str(typ)]={'mean':expected_deg[ix].mean(0).tolist(),
                        'p90':np.quantile(expected_deg[ix],.9,axis=0).tolist(),
                        'maximum':expected_deg[ix].max(0).tolist(),
                        'sorted_expected_degree_by_relation':np.sort(expected_deg[ix],axis=0).tolist(),
                        'reference_maximum':ref[ix].max(0).tolist()}
                result.append(dict(t=inp['t'],pooled_ce=float(ce.sum()/case['space'].total),
                    constant_prior_ce=float(prior_ce.sum()/case['space'].total),count_by_relation_label=size.tolist(),
                    ce_sum_by_relation_label=ce.tolist(),prior_ce_sum_by_relation_label=prior_ce.tolist(),
                    ce_mean_by_relation_label=np.divide(ce,size,out=np.zeros_like(ce),where=size>0).tolist(),
                    missing_cells=np.argwhere(size==0).tolist(),brier_full_slots=brier/case['space'].total,
                    calibration_bins_count_probability_sum_label_sum=bins.tolist(),expected_edges=expected.tolist(),
                    expected_edge_bias=(expected-case['counts']).tolist(),typed_expected_degree=desc,
                    coverage='full active slots exactly once; inactive zero families exact loss=0',
                    analytic_fixed_zero_slots=int(case['space'].counts[~case['active']].sum()),
                    statistic='clean prediction p0, not Bayes q; same-reference denoising only'))
        return dict(times=result,seconds=time.monotonic()-started,execution_mode='ordinary' if not torch.are_deterministic_algorithms_enabled() else 'strict')
    finally:
        c.rng_restore(saved,np.random.default_rng(0),device);model.train(was_training)
