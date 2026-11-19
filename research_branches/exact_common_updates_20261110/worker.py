"""Reuse hybrid-mean baseline; only add four common-direction updates/model."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,copy,signal,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=mod('common_base',ROOT/'research_branches/exact_feedback_updates_20261110/worker.py')
rt,torch,np,g=base.rt,base.torch,base.np,base.g
HardScore,Exact,remaining,loo=base.HardScore,base.Exact,base.remaining,base.loo
solver=mod('common_solver',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py').project_mean
OLD=ROOT/'results/exact_feedback_updates_20261110';OUT=ROOT/'results/exact_common_updates_20261110'
def main():
    index=int(sys.argv[1]);protocol=json.loads((OUT/'protocol.json').read_text());row=protocol['tasks'][index]
    folder=OUT/f"{row['dataset']}_{row['seed']}";folder.mkdir(exist_ok=False);old=OLD/folder.name
    start=time.time();receipt=dict(status='running',updates=0,forwards=0,backwards=0,start_epoch=start)
    def save(n,z):rt.save(folder/n,z)
    def check():
        if time.time()>json.loads((OUT/'window.json').read_text())['science_stop_epoch']:raise TimeoutError('science stop')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('supervisor stop')))
    try:
        rt.setup();assert torch.cuda.is_available() and torch.cuda.device_count()==1;assert '2080 Ti' in torch.cuda.get_device_name(0)
        for p,h in json.loads((OUT/'bindings.json').read_text()).items():
            if p.endswith('.py') or p in (row['checkpoint'],row['data'],row['history']) or str(old) in p:assert rt.sha(p)==h,'binding changed '+p
        z,cfg,case,m,o,e,rng,b=rt.load(row);params=list(m.parameters());names=[n for n,_ in m.named_parameters()]
        fixed=torch.load(old/'evaluation_inputs.pt',map_location='cpu',weights_only=False);baseline=json.loads((old/'updates.json').read_text());baseline=[r for r in baseline if r['arm']=='hybrid']
        history=[json.loads(x) for x in Path(row['history']).read_text().splitlines()]
        inp=fixed['inputs'][0];order=np.argsort(inp['ids']);canonical=inp['ids'][order]
        core=HardScore(case['space'].types,inp['query'][:,order],inp['f'][order],inp['labels'][order],case['n_families']);exact=Exact(core)
        aux=np.random.default_rng(row['seed']+protocol['training_hard_seed_offset']);records=[]
        save('environment.json',g.common.environment('cuda'));save('identity.json',dict(row=row,config=cfg,baseline=str(old)))
        for k in range(4):
            check();tick=time.time();u=b.completed+1
            with b.step(u):
                inp=g.one_input(case,cfg,u,rng);ih=rt.input_hash([inp]);assert ih==history[u-1]['input_sha256']==baseline[k]['input_hash']
                assert inp['ce_weight']==1/case['space'].total
                after_input=g.rng_capture(rng,'cuda');assert g.digest(after_input)==baseline[k]['main_rng_hash']
                o.zero_grad(set_to_none=True);m.train();token=('common_exact',u);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                try:
                    graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);receipt['forwards']+=1
                    g.common.check_prob_device(p,inp['f'],case,m.prior);ix=np.argsort(inp['ids']);assert np.array_equal(inp['ids'][ix],canonical);assert np.array_equal(inp['labels'][ix],core.reference)
                    prob=p.detach().cpu().double().numpy()[ix];margin=(logits[:,1]-logits[:,0])[torch.as_tensor(ix,device='cuda')]
                    gap=float(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()-prob).max());assert gap<2e-6
                    aux_before=g.digest(aux.bit_generator.state);draw=aux.random((64,len(prob)));bits=(draw<prob).astype(np.uint8)
                    assert aux_before==baseline[k]['aux_before_hash'] and g.digest(draw)==baseline[k]['draw_hash'] and g.digest(aux.bit_generator.state)==baseline[k]['aux_after_hash']
                    ts=time.time();vals,dg,_=exact.value_gradient(prob);cost=remaining(core,bits);feedback=np.zeros((5,len(prob)));feedback[[0,2,3]]=dg;feedback[[1,4]]=loo(prob,bits,cost)
                    values=np.zeros(5);values[[0,2,3]]=vals;values[[1,4]]=cost.mean(0);score_seconds=time.time()-ts
                    vectors=[];used=None
                    def vjp(v,keep):
                        nonlocal used
                        gs=torch.autograd.grad(margin,params,grad_outputs=torch.as_tensor(v,device='cuda',dtype=margin.dtype),allow_unused=True,retain_graph=keep);receipt['backwards']+=1
                        live=[x is not None for x in gs]
                        if used is None:used=live
                        assert used==live
                        return torch.cat([(torch.zeros_like(pp) if gg is None else gg).detach().flatten() for pp,gg in zip(params,gs)])
                    for j,v in enumerate(feedback):vectors.append(vjp(v,j<4 or k==0))
                    G=torch.stack(vectors);del vectors;assert torch.isfinite(G).all();mean=G.mean(0)
                    linearity=None
                    if k==0:
                        direct=vjp(feedback.mean(0),False);linearity=float((direct-mean).double().norm()/direct.double().norm().clamp_min(1e-30));assert linearity<1e-3;del direct
                        saved=np.load(old/'train_hybrid_u2001.npz');pgap=float(abs(saved['probability']-prob).max());assert pgap<2e-6
                        save('first_input_comparison.json',dict(probability_max_gap=pgap,parameter_mean_gradient_norm=float(mean.double().norm()),old_parameter_mean_gradient_norm=baseline[0]['gradient_norm'],VJP_linearity_relative_error=linearity,CUDA_recomputation_not_certified_bitwise=True))
                    gram=np.zeros((5,5))
                    for st in range(0,G.shape[1],100000):
                        block=G[:,st:st+100000].double();gram+=(block@block.T).cpu().numpy()
                    ts=time.time();coeff,cert=solver(gram);solver_seconds=time.time()-ts
                    combined=torch.empty_like(mean)
                    for st in range(0,G.shape[1],100000):combined[st:st+100000]=(torch.as_tensor(coeff,device='cuda',dtype=torch.float64)@G[:,st:st+100000].double()).float()
                    assert torch.isfinite(combined).all();gn=float(combined.double().norm());mean_norm=float(mean.double().norm());inner=np.zeros(5)
                    for st in range(0,G.shape[1],100000):inner+=(G[:,st:st+100000].double()@combined[st:st+100000].double()).cpu().numpy()
                    den=np.sqrt(np.maximum(gram.diagonal(),0))*gn;cos_slack=np.divide(inner,den,out=np.zeros(5),where=den>0)
                    assert min(cos_slack)>=-1e-5,'actual FP32 combined gradient fails numerical constraint; no relaxation'
                    offset=0
                    for pp,use in zip(params,used):pp.grad=combined[offset:offset+pp.numel()].view_as(pp).clone() if use else None;offset+=pp.numel()
                    before=[pp.detach().clone() for pp in params]
                    ce=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/case['space'].total
                    b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
                    delta=torch.cat([(pp.detach()-oldp).flatten() for pp,oldp in zip(params,before)]);assert torch.isfinite(delta).all();dn=float(delta.double().norm());actual=np.zeros(5);groups={};offset=0
                    for n,pp in zip(names,params):
                        end=offset+pp.numel();v=G[:,offset:end].double();d=delta[offset:end].double();di=(v@d).cpu().numpy();actual+=di
                        gg=groups.setdefault(base.group(n),dict(actual_directional_derivatives=np.zeros(5),gradient_sq=np.zeros(5),displacement_sq=0.))
                        gg['actual_directional_derivatives']+=di;gg['gradient_sq']+=(v*v).sum(1).cpu().numpy();gg['displacement_sq']+=float(d@d);offset=end
                    for gg in groups.values():
                        for key in ('actual_directional_derivatives','gradient_sq'):gg[key]=gg[key].tolist()
                    actual_den=np.sqrt(np.maximum(gram.diagonal(),0))*dn;actual_cos=np.divide(actual,actual_den,out=np.zeros(5),where=actual_den>0)
                    np.savez_compressed(folder/f'train_common_u{u}.npz',probability=prob,bits=bits,feedback=feedback,values=values,ids=canonical,Gram=gram,coefficients=coeff,combined_inner_products=inner,actual_directional_derivatives=actual)
                    rec=dict(arm='common',update=u,t=inp['t'],input_hash=ih,main_rng_hash=g.digest(after_input),aux_before_hash=aux_before,aux_after_hash=g.digest(aux.bit_generator.state),draw_hash=g.digest(draw),gradient_norm=gn,mean_gradient_norm=mean_norm,correction_relative_norm=float((combined-mean).double().norm()/mean.double().norm().clamp_min(1e-30)),displacement_norm=dn,gradient_dot_displacement=float(combined.double()@delta.double()),group_gradient_norms=np.sqrt(np.maximum(gram.diagonal(),0)).tolist(),coefficients=coeff.tolist(),solver=cert,combined_inner_products=inner.tolist(),combined_cosines=cos_slack.tolist(),actual_directional_derivatives=actual.tolist(),actual_update_cosines=actual_cos.tolist(),modules=groups,score_seconds=score_seconds,solver_seconds=solver_seconds,unweighted_CE=float(ce.detach()),structure_values=values.tolist(),zero_combined_gradient=gn==0)
                finally:ctx.close()
                assert g.digest(g.rng_capture(rng,'cuda'))==g.digest(after_input)
            receipt['updates']+=1;torch.cuda.synchronize();rec.update(seconds=time.time()-tick,completed=b.completed,EMA_updates=e.updates);records.append(rec);save('updates.json',records)
            if k==0:
                elapsed=time.time()-start;estimate=elapsed+3*rec['seconds']+15
                save('calibration.json',dict(elapsed_seconds=elapsed,first_update_seconds=rec['seconds'],estimated_model_seconds=estimate,safety_factor=1.5))
                if time.time()+1.5*(estimate-elapsed+(3-index)*estimate)>json.loads((OUT/'window.json').read_text())['science_stop_epoch']:raise TimeoutError('full panel cannot fit measured budget')
            del G,mean,combined,before,delta,feedback,bits,draw,prob,p,logits,margin,ce
        state=b.checkpoint(z['binding']);rt.checkpoint(folder/'common_complete.pt',dict(state=state,auxiliary_rng=copy.deepcopy(aux.bit_generator.state),diagnostic_only=True,protocol_sha256=rt.sha(OUT/'protocol.json')))
        saved=torch.load(folder/'common_complete.pt',map_location='cpu',weights_only=False);assert g.digest(state)==g.digest(saved['state']);r0=copy.deepcopy(rng);next0=g.one_input(case,cfg,2005,r0)
        b=rt.restore(saved['state'],m,o,e,rng);r1=copy.deepcopy(rng);next1=g.one_input(case,cfg,2005,r1);assert rt.input_hash([next0])==rt.input_hash([next1]);assert g.digest(state)==g.digest(b.checkpoint(z['binding']))
        evaluations=[];before_eval=g.digest(b.checkpoint(z['binding']))
        with b.validation():
            m.eval()
            for j,inp in enumerate(fixed['inputs']):
                check();token=('common_eval',j);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                try:
                    with torch.no_grad():
                        graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                        g.common.check_prob_device(p,inp['f'],case,m.prior);terms=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                    receipt['forwards']+=1;ix=np.argsort(inp['ids']);prob=p.cpu().double().numpy()[ix];y=inp['labels'];f=inp['f'];relations=[]
                    np.savez_compressed(folder/f'eval_common_t{inp["t"]}.npz',probability=prob,terms=terms[ix],labels=y[ix],families=f[ix],ids=canonical,query=inp['query'][:,ix])
                    for family in range(case['n_families']):
                        pos=(f==family)&(y==1);neg=(f==family)&(y==0);relations.append(dict(family=family,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None))
                    evaluations.append(dict(arm='common',t=inp['t'],updates=2004,readout='raw',input_hash=rt.input_hash([inp]),CE=float(terms.sum()/case['space'].total),positive_CE=float(terms[y==1].mean()),negative_CE=float(terms[y==0].mean()),relations=relations))
                finally:ctx.close()
        assert before_eval==g.digest(b.checkpoint(z['binding']));save('evaluation.json',evaluations)
        oldfinal=torch.load(old/'hybrid_complete.pt',map_location='cpu',weights_only=False);assert g.digest(state['rng'])==g.digest(oldfinal['state']['rng']);assert g.digest(saved['auxiliary_rng'])==g.digest(oldfinal['auxiliary_rng'])
        save('checks.json',dict(full_parent_restore=True,save_load_state_exact=True,next_input_after_reload_equal=True,historical_and_baseline_main_inputs_paired=True,auxiliary_uniforms_and_final_RNG_paired=True,evaluation_state_unchanged=True,original_checkpoint_unchanged=rt.sha(row['checkpoint'])==row['checkpoint_sha256'],ordinary_CUDA_recomputation_not_certified_bitwise=True));receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));(folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',receipt)
if __name__=='__main__':main()
