"""Deterministic small engineering states; no scientific generation or old weights."""
import copy
import itertools
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.computational import FrozenTypedContext,GraphInputs
from hsgd.model import TypedSparseDenoiser
from hsgd.kernel import MarginalKernel
from hsgd.runtime import predict,reverse_step
from hsgd.losses import supported_binary_ce

ATOL=2e-6
GRAD_ATOL=2e-6
DECISION_RESPONSE_FLOOR=1e-8


def decision_coordinates(logits,prior):
    """Raw logit gap and ACTUAL supported p0, not two-logit displacement."""
    prior=torch.as_tensor(prior,device=logits.device)
    if logits.ndim!=2 or logits.shape[1]!=2 or prior.shape!=logits.shape[:1]:raise ValueError('decision dimensions')
    margin=logits[:,1]-logits[:,0]
    p=logits.softmax(-1)[:,1]
    p=torch.where(prior==0,torch.zeros_like(p),torch.where(prior==1,torch.ones_like(p),p))
    return margin,p


def decision_response(left,right,prior):
    dl,pl=decision_coordinates(left,prior);dr,pr=decision_coordinates(right,prior)
    return dict(interface_logits_max_abs=float((left-right).abs().max()),
        logit_gap_max_abs=float((dl-dr).abs().max()),p0_max_abs=float((pl-pr).abs().max()))


def decision_mixed_response(c1g1,c0g1,c1g0,c0g0,prior):
    d11,p11=decision_coordinates(c1g1,prior);d01,p01=decision_coordinates(c0g1,prior)
    d10,p10=decision_coordinates(c1g0,prior);d00,p00=decision_coordinates(c0g0,prior)
    return dict(interface_logits_max_abs=float(((c1g1-c0g1)-(c1g0-c0g0)).abs().max()),
        logit_gap_max_abs=float(((d11-d01)-(d10-d00)).abs().max()),
        p0_max_abs=float(((p11-p01)-(p10-p00)).abs().max()))


def fixture():
    case=c.make_case(np.array([0,0,1,1,2,2,2,0]),np.array([[0,1,2,2,3,0,1],[2,2,4,5,6,7,6]]),
        tuple(itertools.combinations_with_replacement(range(3),2)))
    query=np.array([[0,0,1,5],[1,4,3,6]])
    return case,query,case['space'].encode(query)[0]


def make(case,cn,device='cpu',full=False):
    cfg=c.config();cfg['model']['cn_hidden']=32 if cn else 0
    cfg['model'].update(joint_width=128,max_wedges=2000000,max_query_paths=2000000)
    if not full:cfg['model'].update(layers=2,width=16,edge_width=8,global_width=8,heads=2,ff_node=16,ff_edge=8,ff_global=8)
    c.seed_all(1603)
    return c.make_model(case,cfg,device),cfg


def context(model,case,edges=None,tag='fixture'):
    edges=case['clean'] if edges is None else edges
    return FrozenTypedContext(model,case['space'].types,edges,case['space'].encode(edges)[0],tag)


def rng_equal(a,b):
    assert a['generator']==b['generator'] and a['python']==b['python']
    assert a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1])
    assert a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch'],b['torch'])
    assert a['accelerator'] is b['accelerator'] is None


def mechanism():
    case,q,f=fixture();a,_=make(case,False);b,_=make(case,True);a.eval();b.eval();record={}
    for name,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[name]),name
    assert a.joint_head[0].in_features==b.joint_head[0].in_features
    assert a.common_neighbor is None and not any('common_neighbor' in n for n,_ in a.named_parameters())
    with torch.no_grad():
        ctx=context(b,case);graph=ctx.assemble(ctx.transfer_query(ctx.prepare_query(q,f,'fixture'),'fixture'),'fixture')
        logits,features=b.joint_predict(ctx,graph,.03,return_features=True)
        masked=b.joint_predict(ctx,graph,.03,mask_cn=True)
        baseline=predict(a,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.03,details=True)[0]
        assert torch.equal(masked,baseline)
        response=float((logits-masked).abs().max());assert response>1e-7
        record['CN_feature_mask_logit_response']=response
        prior=b.prior[graph.query_families]
        record['CN_feature_mask_decision_response']=decision_response(logits,masked,prior)
        assert record['CN_feature_mask_decision_response']['logit_gap_max_abs']>DECISION_RESPONSE_FLOOR
        assert record['CN_feature_mask_decision_response']['p0_max_abs']>DECISION_RESPONSE_FLOOR
        # Mirror the exact same graph query endpoints, not a fresh query context.
        keys=graph.edges[0]*len(ctx.types)+graph.edges[1]
        reverse_index=torch.searchsorted(keys,graph.edges[1,graph.query_index]*len(ctx.types)+graph.edges[0,graph.query_index])
        reverse=GraphInputs(graph.edges,graph.states,graph.families,graph.query_mask,reverse_index,graph.query_families,graph.attrs)
        swapped=b.joint_predict(ctx,reverse,.03)
        record['endpoint_swap_logit_max_abs']=float((logits-swapped).abs().max())
        assert torch.allclose(logits,swapped,atol=ATOL,rtol=0)
        aggregate=b.common_neighbor.aggregate(ctx,torch.tensor(q),torch.tensor(f),.03)
        # Independent exact common-neighbor enumeration on small Xt only.
        adj={i:set() for i in range(len(ctx.types))}
        for u,v in ctx.old.T:adj[int(u)].add(int(v));adj[int(v)].add(int(u))
        counts=np.zeros((len(f),b.n_types))
        for j,(u,v) in enumerate(q.T):
            for middle in adj[int(u)]&adj[int(v)]:counts[j,ctx.types[middle]]+=1
        assert np.allclose(aggregate[:,32:32+b.n_types],np.log1p(counts),atol=ATOL,rtol=0)
        assert torch.equal(aggregate[:,-1],torch.tensor((counts.sum(1)>0).astype(np.float32)))
        assert torch.equal(aggregate[counts.sum(1)==0],torch.zeros_like(aggregate[counts.sum(1)==0]))
        record['independent_CN_counts']=counts.tolist()
        # More legal Xt edges provide alternative GLOBAL inputs; hold every
        # local feature fixed for this interface-response diagnostic only.
        extra=np.column_stack([case['clean'],[2,6]])
        other=context(b,case,extra,tag='other');changed=features.clone()
        changed[:,b.joint_slices['Xt_statistics']]=other.global_statistics
        global_response=float((b.joint_head(changed)-b.joint_head(features)).abs().max())
        assert global_response>1e-8;record['legal_global_input_only_logit_response']=global_response
        record['legal_global_input_only_decision_response']=decision_response(b.joint_head(changed),b.joint_head(features),prior)
        assert record['legal_global_input_only_decision_response']['logit_gap_max_abs']>DECISION_RESPONSE_FLOOR
        assert record['legal_global_input_only_decision_response']['p0_max_abs']>DECISION_RESPONSE_FLOOR
        without_cn=features.clone();without_cn[:,b.joint_slices['CN']]=0
        changed_without_cn=changed.clone();changed_without_cn[:,b.joint_slices['CN']]=0
        mixed=(b.joint_head(changed)-b.joint_head(changed_without_cn))-(b.joint_head(features)-b.joint_head(without_cn))
        record['CN_global_mixed_finite_difference_max']=float(mixed.abs().max())
        assert record['CN_global_mixed_finite_difference_max']>1e-8
        record['CN_global_decision_mixed_response']=decision_mixed_response(b.joint_head(changed),b.joint_head(changed_without_cn),
            b.joint_head(features),b.joint_head(without_cn),prior)
        assert record['CN_global_decision_mixed_response']['logit_gap_max_abs']>DECISION_RESPONSE_FLOOR
        assert record['CN_global_decision_mixed_response']['p0_max_abs']>DECISION_RESPONSE_FLOOR
        # Isolate neighbor context response in learned content (same CN count).
        other_cn=b.common_neighbor.aggregate(other,torch.tensor(q),torch.tensor(f),.03)
        content_response=float((aggregate[0,:32]-other_cn[0,:32]).abs().max())
        assert content_response>1e-7;record['middle_context_content_response']=content_response
        other.close()
        # Query additions cannot alter Xt-only CN extraction/content.
        q2=np.column_stack([q,[0,6]]);f2=case['space'].encode(q2)[0]
        ctx.assemble(ctx.transfer_query(ctx.prepare_query(q2,f2,'fixture'),'fixture'),'fixture')
        assert torch.equal(aggregate,b.common_neighbor.aggregate(ctx,torch.tensor(q),torch.tensor(f),.03))
        record['query_edges_not_CN_edges']=True
        cnpart=features[:,b.joint_slices['CN']]
        head=copy.deepcopy(b.joint_head)
        for p in head.parameters():p.zero_()
        j=b.joint_slices['CN'].start+32
        head[0].weight[0,j]=1.;head[2].weight[1,0]=-1.
        low=torch.zeros_like(features[:1]);high=low.clone();low[0,j]=np.log(2);high[0,j]=np.log(3)
        assert head(high).softmax(-1)[0,1]<head(low).softmax(-1)[0,1]
        record['no_architectural_positive_count_monotonicity']=True
        quant=lambda x:[float(v) for v in torch.quantile(x.flatten(),torch.tensor([0.,.5,.9,.99,1.]))]
        record['feature_abs_quantiles']=quant(features.abs());record['logit_quantiles']=quant(logits)
        p=logits.softmax(-1)[:,1];record['probability_saturation_fraction']=float(((p<1e-6)|(p>1-1e-6)).float().mean())
        ctx.close()
    # Type-preserving permutation of endpoints, same expected query order.
    perm=np.array([1,0,3,2,5,4,6,7]);remap=lambda e:np.sort(perm[e],axis=0)
    with torch.no_grad():
        left=predict(b,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.03,details=True)[0]
        right=predict(b,case['space'].types,remap(case['clean']),case['space'].encode(case['clean'])[0],remap(q),f,.03,details=True)[0]
    record['renumber_logit_max_abs']=float((left-right).abs().max());assert torch.allclose(left,right,atol=ATOL,rtol=0)
    # Loss really reaches phi AND the common nonlinear head.
    b.train();logits,p=predict(b,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.03,details=True)
    loss=supported_binary_ce(logits,torch.tensor([0.,1.,0.,0.]),b.prior[torch.tensor(f)]).mean();loss.backward()
    record['phi_gradient_max']=max(float(x.grad.abs().max()) for x in b.common_neighbor.parameters() if x.grad is not None)
    assert record['phi_gradient_max']>0 and b.joint_head[0].weight.grad.abs().max()>0
    record['global_head_input_gradient_max']=float(b.joint_head[0].weight.grad[:,b.joint_slices['backbone_global']].abs().max())
    record['global_update_parameter_gradient_max']=float(b.backbone.tf_layers[0].self_attn.y_out[0].weight.grad.abs().max())
    assert record['global_head_input_gradient_max']>0 and record['global_update_parameter_gradient_max']>0
    assert all(torch.isfinite(x.grad).all() for x in b.parameters() if x.grad is not None)
    # One implementation serves direct training and StepPredictor sampling.
    b.eval();kernel=MarginalKernel(case['prior'],c.schedule(1000));state=np.zeros(case['space'].total,dtype=np.uint8);state[case['clean_ids']]=1
    rng=np.random.default_rng(719);next_state,row=reverse_step(b,case,state,kernel,30,3,17,rng,'cpu',trace=True)
    plan=c.plan_for(case,[17,30,61],3);direct=np.zeros(len(state))
    with torch.no_grad():
        for r in range(3):
            ids,fam,query=plan.block(r)
            if len(ids):direct[ids]=predict(b,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],query,fam,.03).numpy()
    record['train_sample_probability_max_abs']=float(np.max(abs(direct-row['trace']['p0'])))
    assert record['train_sample_probability_max_abs']<=ATOL
    assert np.array_equal(state, np.isin(np.arange(len(state)),case['clean_ids']).astype(np.uint8))
    assert row['exact_coverage']
    state2=np.zeros_like(state);rng2=np.random.default_rng(719);rng_state=c.rng_save(rng2,'cpu')
    try:reverse_step(b,case,state2,kernel,30,3,17,rng2,'cpu',interrupt_block=1)
    except InterruptedError:pass
    else:raise AssertionError('partial step unexpectedly committed')
    rng_equal(rng_state,c.rng_save(rng2,'cpu'));assert not state2.any()
    # Full tiny T4 chain is engineering only, every reverse step frozen.
    k4=MarginalKernel(case['prior'],c.schedule(4));rs=np.random.default_rng(917)
    tiny=(rs.random(len(state))<np.repeat(case['prior'],case['space'].counts)).astype(np.uint8)
    for t in range(4,0,-1):tiny,_=reverse_step(b,case,tiny,k4,t,3,17,rs,'cpu')
    assert np.isin(tiny,[0,1]).all();record['small_engineering_full_chain_steps']=4
    # Fresh native save/load (not historical CN2 compatibility).
    clone,_=make(case,True);clone.load_state_dict(b.state_dict(),strict=True);clone.eval()
    with torch.no_grad():
        restored=predict(clone,case['space'].types,case['clean'],case['space'].encode(case['clean'])[0],q,f,.03,details=True)[0]
    assert torch.equal(left,restored)
    record.update(status='passed',atol=ATOL,scope='random-weight small CPU engineering; response is NOT learned coordination',
        new_scientific_graphs=0,GPU_use=0,old_weights_loaded=False)
    return record


def diagnostics(model,case,inp,T):
    """Same frozen query in full-vs-CN-mask diagnostics; NOT additive deltas."""
    model.eval();xt=np.zeros(case['space'].total,dtype=np.uint8)
    ff,ll=case['space'].encode(inp['edges']);xt[case['space'].offsets[ff]+ll]=1
    ctx=context(model,case,inp['edges']);token='fixture'
    try:
        graph=ctx.assemble(ctx.transfer_query(ctx.prepare_query(inp['query'],inp['f'],token),token),token)
        with torch.no_grad():
            logits,features=model.joint_predict(ctx,graph,inp['t']/T,return_features=True)
            masked=model.joint_predict(ctx,graph,inp['t']/T,mask_cn=True)
        p=logits.softmax(-1)[:,1];pi=model.prior[torch.tensor(inp['f'],device=p.device)]
        p=torch.where(pi==0,torch.zeros_like(p),torch.where(pi==1,torch.ones_like(p),p))
        prob=p.cpu().numpy().astype(float);q=MarginalKernel(case['prior'],c.schedule(T)).reverse(prob,xt[inp['ids']],inp['f'],inp['t']-1,inp['t'])
        labels=np.asarray(inp['labels']);item=supported_binary_ce(logits,torch.tensor(labels,device=p.device),pi).cpu().numpy()
        # Diagnostic grouping must use the SAME actual Xt counts for both arms.
        # This offline count is not supplied to the disabled baseline branch.
        from hsgd.common_neighbor import Wedges
        wedges=ctx.cn_wedges if ctx.cn_wedges is not None else Wedges(ctx)
        qid,_,_,_=wedges.lookup(torch.tensor(inp['query'],device=ctx.device))
        cncount=torch.bincount(qid,minlength=len(prob)).cpu().numpy()
        present=xt[inp['ids']].astype(bool)
        groups=[]
        for f in range(case['n_families']):
            for sign in (0,1):
                mask=(inp['f']==f)&(labels==sign);n=int(mask.sum())
                groups.append(dict(family=f,clean_label=sign,count=n,CE=float(item[mask].mean()) if n else None,
                    p_present_mean=float(prob[mask].mean()) if n else None,
                    expected_birth=float(q[mask&~present].sum()),expected_death=float((1-q[mask&present]).sum()),
                    expected_net=float(q[mask&~present].sum()-(1-q[mask&present]).sum())))
        buckets=[]
        for lo,hi in [(0,1),(1,2),(2,4),(4,16),(16,float('inf'))]:
            sel=(cncount>=lo)&(cncount<hi)
            birth=float(q[sel&~present].sum());death=float((1-q[sel&present]).sum())
            buckets.append(dict(CN_range=[lo,None if np.isinf(hi) else hi],count=int(sel.sum()),
                p_present_mean=float(prob[sel].mean()) if sel.any() else None,expected_birth=birth,expected_death=death,expected_net=birth-death))
        return dict(t=inp['t'],time_group='low' if inp['t']<=100 else 'middle' if inp['t']<=750 else 'high',
            scope='query-only expected counts; not extrapolated to whole graph',relation_label_CE=groups,CN_buckets=buckets,
            CN_count_source='exact Xt topology for BOTH arms; diagnostic-only when baseline branch disabled',
            feature_abs_quantiles=np.quantile(features.abs().cpu().numpy(),[0,.5,.9,.99,1]).tolist(),
            logits_quantiles=np.quantile(logits.cpu().numpy(),[0,.5,.9,.99,1]).tolist(),
            saturation_fraction=float(np.mean((prob<1e-6)|(prob>1-1e-6))),
            CN_mask_logit_difference_max=float((logits-masked).abs().max()),
            CN_mask_decision_response=decision_response(logits,masked,pi),
            masking_caveat='counterfactual feature masking only, not independent trained baseline or additive decomposition')
    finally:ctx.close()
