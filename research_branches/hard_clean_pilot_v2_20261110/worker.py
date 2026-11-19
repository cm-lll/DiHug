"""One bounded, isolated CE or hard-clean-risk pilot; original reverse kernel untouched."""
import time, signal, traceback, copy
from runtime import *
from objective import HardScore, loo_gradient, solve_gram
from exact_absent import ExactAbsentTriangles


def main():
    out=Path(sys.argv[1]); index=int(sys.argv[2]); mode=sys.argv[3]
    protocol=json.loads((out/'protocol.json').read_text()); row=protocol['tasks'][index]
    folder=out/('gate_'+row['task'] if mode=='gate' else row['task']);folder.mkdir(exist_ok=False)
    start=time.time(); stop=[False]; status=dict(status='loading',updates=0,forwards=0,backwards=0,task=row['task'])
    save(folder/'allocation.json',dict(start_epoch=start,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),mode=mode))
    for s in (signal.SIGTERM,signal.SIGINT): signal.signal(s,lambda *_:stop.__setitem__(0,True))
    b=None; aux=None; binding=None
    def check():
        limits=json.loads((out/'window.json').read_text())
        if stop[0] or (out/'STOP').exists() or time.time()>min(limits['science_stop_epoch']-30,start+protocol['budgets']['worker_minutes']*60-30):
            raise TimeoutError('bounded science/GPU stop')
    try:
        check();setup()
        for path,h in json.loads((out/'bindings.json').read_text()).items():
            # Source and this task's input assets; other parent hashes checked by coordinator.
            if path.endswith('.py') or path in (row['checkpoint'],row['data'],row['history']):
                assert sha(path)==h,'asset changed '+path
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert '2080 Ti' in torch.cuda.get_device_name(0)
        z,cfg,case,m,o,e,rng,b=load(row);binding=z['binding'];params=list(m.parameters())
        save(folder/'environment.json',g.common.environment('cuda'))
        save(folder/'identity.json',dict(row=row,parent_updates=z['updates'],config=cfg,
              normal_mode=True,scientific_change='loss only; parent AdamW inherited',restore_bitwise=True))
        history=[json.loads(s) for s in Path(row['history']).read_text().splitlines()]
        aux=np.random.default_rng(row['seed']+protocol['auxiliary_seed_offset'])
        # Probe construction uses a private generator; snapshot ALL global streams as well.
        saved_rng=g.rng_capture(rng,'cuda')
        fixed=[]
        for j,t in enumerate(protocol['evaluation_times']):
            er=np.random.default_rng(row['seed']+protocol['evaluation_seed_offset']+j)
            inp=g.one_input(case,cfg,700000+j,er,force_t=t)
            fixed.append(inp)
        g.rng_restore(saved_rng,rng,'cuda')
        order=np.argsort(fixed[0]['ids']); canonical_ids=fixed[0]['ids'][order]
        core=HardScore(case['space'].types,fixed[0]['query'][:,order],fixed[0]['f'][order],fixed[0]['labels'][order],case['n_families'])
        analytic=ExactAbsentTriangles(core)
        assert int(core.reference.sum())==int(case['counts'].sum())
        save(folder/'score_definition.json',dict(names=core.names,active=core.active,typed_triples=core.triples,
              reference={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in core.ref.items()},
              inactive_zero_support_slots=int(case['space'].total-len(canonical_ids)),
              joint_denominator=max(1,core.n-1),component_denominator=max(1,core.n-1)))
        checkpoint(folder/'evaluation_inputs.pt',dict(inputs=cpu_copy(fixed),hashes=[input_hash([x]) for x in fixed]))
        records=[]; evaluations=[]

        def persist(name):
            state=b.checkpoint(binding)
            checkpoint(folder/name,dict(state=state,auxiliary_rng=copy.deepcopy(aux.bit_generator.state),
                         pilot=True,task=row,protocol_sha256=sha(out/'protocol.json')))
            return state

        def one():
            check();u=b.completed+1;tick=time.perf_counter();aux_before=copy.deepcopy(aux.bit_generator.state)
            try:
                with b.step(u):
                    inp=g.one_input(case,cfg,u,rng);ih=input_hash([inp])
                    assert ih==history[u-1]['input_sha256'],'historical main input mismatch'
                    input_rng=g.digest(g.rng_capture(rng,'cuda'))
                    if row['arm']=='CE':
                        rec=g.update(m,o,case,inp,1000,'cuda',e,u-1,transaction=b)
                        status['forwards']+=1;status['backwards']+=1
                    else:
                        expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
                        assert np.array_equal(np.sort(inp['ids']),expected) and inp['ce_weight']==1/case['space'].total
                        o.zero_grad(set_to_none=True);m.train();token=('hard_clean',u)
                        ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                        try:
                            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True);status['forwards']+=1
                            g.common.check_prob_device(p,inp['f'],case,m.prior)
                            order=np.argsort(inp['ids']);assert np.array_equal(inp['ids'][order],canonical_ids)
                            assert np.array_equal(inp['query'][:,order],core.q) and np.array_equal(inp['labels'][order],core.reference)
                            prob=p.detach().cpu().double().numpy()[order]
                            margin=(logits[:,1]-logits[:,0])[torch.as_tensor(order,device='cuda')]
                            active=(prob>0)&(prob<1)
                            perr=float(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()[active]-prob[active]).max()) if active.any() else 0.
                            assert perr<2e-6,'margin derivative/probability contract'
                            bits=(aux.random((protocol['train_hard_draws'],len(prob)))<prob).astype(np.uint8)
                            score_tick=time.perf_counter();costs=core.batch(bits);all_dg=loo_gradient(prob,bits,costs)
                            exact_absent_value,exact_absent_gradient=analytic.value_gradient(prob)
                            all_dg[3]=exact_absent_gradient;dg=all_dg[core.active]
                            dg[:,~active]=0.;score_seconds=time.perf_counter()-score_tick
                            vectors=[];used=None
                            for j,v in enumerate(dg):
                                grads=torch.autograd.grad(margin,params,grad_outputs=torch.as_tensor(v,device='cuda',dtype=margin.dtype),retain_graph=j<len(dg)-1,allow_unused=True)
                                this=[x is not None for x in grads]
                                if used is None:used=this
                                assert used==this,'gradient support differs across objectives'
                                vectors.append(torch.cat([(torch.zeros_like(p0) if x is None else x).detach().flatten() for p0,x in zip(params,grads)]))
                                status['backwards']+=1
                            G=torch.stack(vectors);assert torch.isfinite(G).all()
                            gram=(G.double()@G.double().T).cpu().numpy();lam,residual=solve_gram(gram)
                            combined=torch.as_tensor(lam,device='cuda',dtype=G.dtype)@G
                            gn=float(combined.double().norm())
                            if gn<=1e-12:
                                save(folder/f'zero_gradient_u{u}.json',dict(costs=costs.tolist(),gradient_norms=np.sqrt(np.maximum(gram.diagonal(),0)).tolist(),coefficients=lam.tolist(),reason='no usable sampled common direction; no objective omitted'))
                                raise RuntimeError('sampled MGDA direction zero; affected trajectory stopped')
                            offset=0
                            for p0,use in zip(params,used):
                                p0.grad=combined[offset:offset+p0.numel()].view_as(p0).clone() if use else None;offset+=p0.numel()
                            ce=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).sum()/case['space'].total
                            assert torch.isfinite(ce)
                            b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
                            rec=dict(loss=None,unweighted_CE=float(ce.detach()),hard_cost_mean=costs.mean(0).tolist(),hard_cost_std=costs.std(0,ddof=1).tolist(),
                                     gradient_norm=gn,individual_gradient_norms=np.sqrt(np.maximum(gram.diagonal(),0)).tolist(),
                                     coefficients=lam.tolist(),Gram=gram.tolist(),KKT=residual,score_seconds=score_seconds,
                                     CE_training_coefficient=0,probability_contract_error=perr,
                                     exact_absent_triangle_risk=exact_absent_value,
                                     caveat='MGDA of estimates is biased/nonlinear; AdamW need not preserve common descent')
                        finally:ctx.close()
                    assert g.digest(g.rng_capture(rng,'cuda'))==input_rng,'hard sampling/gradient altered main RNG'
                status['updates']+=1;torch.cuda.synchronize()
                rec.update(update=u,t=inp['t'],input_sha256=ih,main_rng_sha256=input_rng,seconds=time.perf_counter()-tick,
                           aux_rng_sha256=g.digest(aux.bit_generator.state),completed=b.completed,ema_updates=e.updates)
                records.append(rec)
                with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
                save(folder/'progress.json',dict(status,at_epoch=time.time(),last=rec))
                return rec
            except BaseException:
                if b.resumable:aux.bit_generator.state=aux_before
                raise

        def evaluate(offset):
            check();before=g.digest(g.rng_capture(rng,'cuda'));aux_state=g.digest(aux.bit_generator.state)
            with b.validation():
                raw=cpu_copy(m.state_dict())
                try:
                    for readout in protocol['readouts']:
                        m.load_state_dict(raw if readout=='raw' else e.readout_state_dict(m),strict=True);m.eval()
                        for j,inp in enumerate(fixed):
                            check();token=('independent_eval',offset,readout,j)
                            ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                            try:
                                with torch.no_grad():
                                    graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                                    g.common.check_prob_device(p,inp['f'],case,m.prior)
                                    terms=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                                status['forwards']+=1;order=np.argsort(inp['ids']);prob=p.cpu().numpy()[order]
                                er=np.random.default_rng(row['seed']+protocol['evaluation_seed_offset']+100+j)
                                bits=(er.random((protocol['evaluation_hard_draws'],len(prob)))<prob).astype(np.uint8)
                                costs=[];stats=[]
                                for bb in bits:
                                    c,v=core.costs(bb);costs.append(c)
                                    stats.append(dict(edges=int(bb.sum()),triangles=int(v['triangles'].sum()),typed_triangles=v['triangles'].tolist(),
                                        components=v['components'],LCC=v['lcc']/core.n,isolated=v['isolated'],hub=v['max_degree']))
                                y=inp['labels'];relations=[]
                                for f in range(case['n_families']):
                                    ix=inp['f']==f;pos=ix&(y==1);neg=ix&(y==0)
                                    relations.append(dict(family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),
                                        positive_CE=float(terms[pos].mean()) if pos.any() else None,negative_CE=float(terms[neg].mean()) if neg.any() else None,
                                        expected_edges=float(prob[core.f==f].sum()),reference_edges=int(core.ref['counts'][f])))
                                row_eval=dict(offset=offset,updates=b.completed,readout=readout,t=inp['t'],input_sha256=input_hash([inp]),
                                        CE=float(terms.sum()/case['space'].total),positive_CE=float(terms[y==1].mean()),negative_CE=float(terms[y==0].mean()),
                                        hard_cost_mean=np.mean(costs,0).tolist(),hard_cost_std=np.std(costs,axis=0,ddof=1).tolist(),
                                        samples=stats,relations=relations,interpretation='conditional clean prediction, NOT reverse diffusion output')
                                evaluations.append(row_eval)
                                np.savez_compressed(folder/f'pred_{offset}_{readout}_{inp["t"]}.npz',probability=prob,bits=bits,costs=costs,ids=canonical_ids)
                                save(folder/'evaluation.json',evaluations)
                            finally:ctx.close()
                finally:m.load_state_dict(raw,strict=True)
            assert g.digest(g.rng_capture(rng,'cuda'))==before and g.digest(aux.bit_generator.state)==aux_state

        if mode=='gate':
            one();saved=persist('gate_complete.pt');saved_aux=copy.deepcopy(aux.bit_generator.state)
            r0=copy.deepcopy(rng);next0=g.one_input(case,cfg,b.completed+1,r0)
            draw0=np.random.default_rng();draw0.bit_generator.state=copy.deepcopy(saved_aux);a0=draw0.random((2,7))
            state=torch.load(folder/'gate_complete.pt',map_location='cpu');b=restore(state['state'],m,o,e,rng)
            aux.bit_generator.state=state['auxiliary_rng'];r1=copy.deepcopy(rng);next1=g.one_input(case,cfg,b.completed+1,r1)
            draw1=np.random.default_rng();draw1.bit_generator.state=copy.deepcopy(aux.bit_generator.state)
            assert input_hash([next0])==input_hash([next1]) and np.array_equal(a0,draw1.random((2,7)))
            one();left=b.checkpoint(binding)
            b=restore(saved,m,o,e,rng);aux.bit_generator.state=copy.deepcopy(saved_aux)
            one();right=b.checkpoint(binding)
            cmp=g.compare_state(left,right,cfg['tolerances']['fp32_atol'],cfg['tolerances']['fp32_rtol'])
            # Historical ordinary-GPU recomputation variance is retained, not relabelled exact.
            save(folder/'gate.json',dict(passed=True,load_bitwise=True,next_main_input_exact=True,next_aux_draw_exact=True,
                  recomputation=cmp,measured_update_seconds=[r['seconds'] for r in records],
                  FP32_recomputation_not_certified_as_exact=not cmp['bitwise_equal']))
        else:
            evaluate(0)
            for k in range(1,protocol['updates_each']+1):
                one()
                if k%32==0:persist('latest_complete.pt')
                if k in protocol['evaluation_offsets']:evaluate(k)
                if k==4:
                    estimate=float(np.mean([r['seconds'] for r in records]))*(protocol['updates_each']-k)*1.5+120
                    end=json.loads((out/'window.json').read_text())['science_stop_epoch']
                    if time.time()+estimate>end:raise TimeoutError('throughput cannot cover complete frozen pilot')
            persist('final_complete.pt')
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        status['status']='complete'
    except BaseException as exc:
        status.update(status='failed_or_stopped',error=repr(exc),traceback=traceback.format_exc())
        if b is not None and binding is not None:
            try:
                checkpoint(folder/'stopped_complete.pt',dict(state=b.checkpoint(binding),auxiliary_rng=copy.deepcopy(aux.bit_generator.state),pilot=True,task=row))
            except BaseException as se:save(folder/'nonresumable.json',dict(error=repr(se),use_previous_complete_only=True))
    finally:
        status.update(end_epoch=time.time(),seconds=time.time()-start,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0)
        save(folder/'receipt.json',status)
        save(folder/'release.json',dict(start_epoch=start,end_epoch=time.time(),occupied_seconds=time.time()-start,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        print(json.dumps(status),flush=True)
    return 0 if status['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())

