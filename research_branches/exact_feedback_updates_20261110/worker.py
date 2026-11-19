"""Four temporary updates per estimator from the same full parent boundary.
Not a formal training entry; no reverse generation. Hard draws are conditional
clean predictions used to estimate the unchanged five losses.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,traceback,importlib.util,copy,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/exact_feedback_updates_20261110'
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rt=mod('ef_runtime',ROOT/'research_branches/hard_clean_pilot_v2_20261110/runtime.py')
torch,np,g=rt.torch,rt.np,rt.g
HardScore=mod('ef_objective',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py').HardScore
loo=mod('ef_loo',ROOT/'research_branches/hard_clean_zero_safe_20261110/combine.py').loo_gradient
Exact=mod('ef_exact',ROOT/'research_branches/exact_hard_moments_20261110/moments.py').ExactHardMoments
remaining=mod('ef_remaining',ROOT/'research_branches/exact_hard_moments_20261110/remaining.py').remaining_costs
def group(n):
    if n.startswith('backbone.'):return 'shared_backbone'
    if n.startswith('common_neighbor.'):return 'CN_exclusive'
    if n.startswith('joint_head.'):return 'joint_head'
    if n.startswith('branch.degree'):return 'degree_branch'
    return 'other'
def main():
    idx=int(sys.argv[1]);protocol=json.loads((OUT/'protocol.json').read_text());row=protocol['tasks'][idx]
    folder=OUT/f"{row['dataset']}_{row['seed']}";folder.mkdir(exist_ok=False)
    start=time.time();receipt=dict(status='running',updates=0,forwards=0,backwards=0,start_epoch=start)
    def save(n,z):rt.save(folder/n,z)
    def check():
        window=json.loads((OUT/'window.json').read_text())
        if time.time()>window['science_stop_epoch']:raise TimeoutError('science stop')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('supervisor stop')))
    try:
        rt.setup();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert '2080 Ti' in torch.cuda.get_device_name(0)
        for p,h in json.loads((OUT/'bindings.json').read_text()).items():
            if p.endswith('.py') or p in (row['checkpoint'],row['data'],row['history']):assert rt.sha(p)==h,'binding changed '+p
        z,cfg,case,m,o,e,rng,b=rt.load(row);params=list(m.parameters());names=[n for n,_ in m.named_parameters()]
        save('environment.json',g.common.environment('cuda'));save('identity.json',dict(row=row,config=cfg))
        history=[json.loads(x) for x in Path(row['history']).read_text().splitlines()]
        original_rng=g.rng_capture(rng,'cuda');fixed=[]
        for j,t in enumerate(protocol['evaluation_times']):
            fixed.append(g.one_input(case,cfg,910000+j,np.random.default_rng(row['seed']+protocol['evaluation_seed_offset']+j),force_t=t))
        g.rng_restore(original_rng,rng,'cuda')
        rt.checkpoint(folder/'evaluation_inputs.pt',dict(inputs=rt.cpu_copy(fixed),hashes=[rt.input_hash([x]) for x in fixed]))
        order=np.argsort(fixed[0]['ids']);core=HardScore(case['space'].types,fixed[0]['query'][:,order],fixed[0]['f'][order],fixed[0]['labels'][order],case['n_families']);exact=Exact(core)
        canonical=fixed[0]['ids'][order]
        oldfixed=torch.load(ROOT/f"results/hard_clean_pilot_v2_20261110/{row['dataset']}_{row['seed']}_HARD_MGDA/evaluation_inputs.pt",map_location='cpu',weights_only=False)
        assert not set(rt.input_hash([x]) for x in fixed)&set(oldfixed['hashes']),'new evaluation inputs duplicated old panel'
        save('score_definition.json',dict(names=core.names,reference={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in core.ref.items()},legal_slots=len(canonical),full_schema_slots=case['space'].total,triples=core.triples))
        records=[];evals=[];first_grad={};first_delta={};initial_probs={}
        def evaluate(arm):
            check();before=g.digest(b.checkpoint(z['binding']));tick=time.time()
            with b.validation():
                m.eval()
                for j,inp in enumerate(fixed):
                    check();token=('exact_feedback_eval',arm,j);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                    try:
                        with torch.no_grad():
                            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                            g.common.check_prob_device(p,inp['f'],case,m.prior)
                            terms=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                        receipt['forwards']+=1;ix=np.argsort(inp['ids']);prob=p.cpu().double().numpy()[ix]
                        assert np.array_equal(inp['ids'][ix],canonical) and np.isfinite(prob).all()
                        np.savez_compressed(folder/f'eval_{arm}_t{inp["t"]}.npz',probability=prob,terms=terms[ix],labels=inp['labels'][ix],families=inp['f'][ix],ids=canonical,query=inp['query'][:,ix])
                        relations=[];y=inp['labels'];f=inp['f']
                        for k in range(case['n_families']):
                            pos=(f==k)&(y==1);neg=(f==k)&(y==0)
                            relations.append(dict(family=k,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None))
                        evals.append(dict(arm=arm,t=inp['t'],readout='raw',updates=b.completed,input_hash=rt.input_hash([inp]),CE=float(terms.sum()/case['space'].total),positive_CE=float(terms[y==1].mean()),negative_CE=float(terms[y==0].mean()),relations=relations))
                    finally:ctx.close()
            assert g.digest(b.checkpoint(z['binding']))==before,'evaluation changed state'
            save('evaluation.json',evals);return time.time()-tick
        baseline_eval_seconds=evaluate('before')
        for arm in protocol['arms']:
            b=rt.restore(z,m,o,e,rng);aux=np.random.default_rng(row['seed']+protocol['training_hard_seed_offset'])
            for k in range(protocol['updates_each']):
                check();u=b.completed+1;tick=time.time()
                with b.step(u):
                    inp=g.one_input(case,cfg,u,rng);ih=rt.input_hash([inp]);assert ih==history[u-1]['input_sha256']
                    assert inp['ce_weight']==1/case['space'].total
                    assert ih not in [rt.input_hash([x]) for x in fixed]
                    after_input=g.rng_capture(rng,'cuda');aux_before=copy.deepcopy(aux.bit_generator.state)
                    o.zero_grad(set_to_none=True);m.train();token=('exact_feedback_train',arm,u)
                    ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                    try:
                        graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);receipt['forwards']+=1
                        g.common.check_prob_device(p,inp['f'],case,m.prior)
                        ix=np.argsort(inp['ids']);assert np.array_equal(inp['ids'][ix],canonical)
                        assert np.array_equal(inp['labels'][ix],core.reference)
                        prob=p.detach().cpu().double().numpy()[ix];margin=(logits[:,1]-logits[:,0])[torch.as_tensor(ix,device='cuda')]
                        gap=float(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()-prob).max());assert gap<2e-6
                        draw=aux.random((protocol['train_hard_draws'],len(prob)));bits=(draw<prob).astype(np.uint8)
                        score_start=time.time()
                        if arm=='sampled':
                            cost=core.batch(bits);feedback=loo(prob,bits,cost);values=cost.mean(0)
                        else:
                            vals,grads,_=exact.value_gradient(prob);cost=remaining(core,bits);feedback=np.zeros((5,len(prob)));feedback[[0,2,3]]=grads;feedback[[1,4]]=loo(prob,bits,cost);values=np.zeros(5);values[[0,2,3]]=vals;values[[1,4]]=cost.mean(0)
                        score_seconds=time.time()-score_start
                        assert np.isfinite(feedback).all() and np.isfinite(values).all()
                        ce=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/case['space'].total
                        margin.backward(torch.as_tensor(feedback.mean(0),device='cuda',dtype=margin.dtype));receipt['backwards']+=1
                        grads=torch.cat([(torch.zeros_like(pp) if pp.grad is None else pp.grad).detach().flatten() for pp in params]);assert torch.isfinite(grads).all()
                        before=[pp.detach().clone() for pp in params]
                        b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
                        delta=torch.cat([(pp.detach()-old).flatten() for pp,old in zip(params,before)]);assert torch.isfinite(delta).all()
                        groups={};offset=0
                        for n,pp in zip(names,params):
                            sub=grads[offset:offset+pp.numel()].double();dd=delta[offset:offset+pp.numel()].double();offset+=pp.numel();gg=groups.setdefault(group(n),dict(gradient_sq=0.,displacement_sq=0.,gradient_dot_displacement=0.))
                            gg['gradient_sq']+=float(sub@sub);gg['displacement_sq']+=float(dd@dd);gg['gradient_dot_displacement']+=float(sub@dd)
                        if k==0:
                            first_grad[arm]=grads.cpu();first_delta[arm]=delta.cpu();initial_probs[arm]=prob.copy()
                        np.savez_compressed(folder/f'train_{arm}_u{u}.npz',probability=prob,bits=bits,feedback=feedback,values=values,ids=canonical)
                        rec=dict(arm=arm,update=u,t=inp['t'],input_hash=ih,main_rng_hash=g.digest(after_input),aux_before_hash=g.digest(aux_before),aux_after_hash=g.digest(aux.bit_generator.state),draw_hash=g.digest(draw),gradient_norm=float(grads.double().norm()),displacement_norm=float(delta.double().norm()),gradient_dot_displacement=float(grads.double()@delta.double()),groups=groups,unweighted_CE=float(ce.detach()),structure_values=values.tolist(),score_seconds=score_seconds,probability_contract_error=gap)
                    finally:ctx.close()
                    assert g.digest(g.rng_capture(rng,'cuda'))==g.digest(after_input),'diagnostic changed main RNG'
                receipt['updates']+=1;torch.cuda.synchronize();rec.update(seconds=time.time()-tick,completed=b.completed,EMA_updates=e.updates);records.append(rec);save('updates.json',records)
                if arm=='sampled' and k==0:
                    elapsed=time.time()-start
                    projected=elapsed+3*baseline_eval_seconds+7*rec['seconds']+20
                    save('calibration.json',dict(elapsed_seconds=elapsed,baseline_six_eval_seconds=baseline_eval_seconds,first_update_seconds=rec['seconds'],model_estimated_seconds=projected,safety_factor=1.5))
                    remaining_models=4-idx
                    if time.time()+1.5*(projected-elapsed+(remaining_models-1)*projected)>json.loads((OUT/'window.json').read_text())['science_stop_epoch']:
                        raise TimeoutError('full panel no longer fits measured budget')
                del grads,delta,before,feedback,draw,bits,prob,logits,p,margin,ce
            state=b.checkpoint(z['binding']);path=folder/f'{arm}_complete.pt';rt.checkpoint(path,dict(state=state,auxiliary_rng=copy.deepcopy(aux.bit_generator.state),diagnostic_only=True,protocol_sha256=rt.sha(OUT/'protocol.json')))
            on_disk=torch.load(path,map_location='cpu',weights_only=False);assert g.digest(state)==g.digest(on_disk['state'])
            r0=copy.deepcopy(rng);next0=g.one_input(case,cfg,b.completed+1,r0)
            b=rt.restore(on_disk['state'],m,o,e,rng);r1=copy.deepcopy(rng);next1=g.one_input(case,cfg,b.completed+1,r1)
            assert rt.input_hash([next0])==rt.input_hash([next1]);assert g.digest(state)==g.digest(b.checkpoint(z['binding']))
            evaluate(arm)
        def cosine(a,b):
            den=float(a.double().norm()*b.double().norm());return float(a.double()@b.double()/den) if den else None
        save('first_step_comparison.json',dict(parameter_gradient_cosine=cosine(first_grad['sampled'],first_grad['hybrid']),actual_displacement_cosine=cosine(first_delta['sampled'],first_delta['hybrid']),initial_probability_max_gap=float(abs(initial_probs['sampled']-initial_probs['hybrid']).max()),note='CE-parent AdamW moments retained; displacement includes momentum and weight decay; ordinary CUDA recomputation not certified bitwise'))
        for k in range(protocol['updates_each']):
            a,c=records[k],records[protocol['updates_each']+k]
            for key in ('input_hash','main_rng_hash','draw_hash','aux_before_hash','aux_after_hash'):assert a[key]==c[key],key+' pairing failed'
        assert rt.sha(row['checkpoint'])==row['checkpoint_sha256'];receipt['status']='completed'
        save('checks.json',dict(full_parent_restore=True,saved_state_bitwise=True,next_input_after_reload_equal=True,main_inputs_and_rng_paired=True,auxiliary_uniforms_paired=True,evaluation_state_unchanged=True,original_checkpoint_unchanged=True,raw_only=True,ordinary_recomputation_not_bitwise_certified=True))
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));(folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(end_epoch=time.time(),seconds=time.time()-start,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0);save('receipt.json',receipt)
if __name__=='__main__':main()
