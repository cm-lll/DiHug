"""Four saved failure boundaries, one temporary update each. No reverse chain.

All scores, K=16 training draws, K=32 paired evaluation draws and six old
development inputs are reused. No CE training term, no gradient clipping,
no change to AdamW/EMA or normal GPU nondeterminism policy.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, signal, copy, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_pilot_v2_20261110'))
from runtime import *
from objective import HardScore, solve_gram
from exact_absent import ExactAbsentTriangles
sys.path.insert(0,str(HERE))
from combine import project_mean, loo_gradient

OUT=ROOT/'results/hard_clean_zero_safe_20261110'
OLD=ROOT/'results/hard_clean_pilot_v2_20261110'


def prepare():
    protocol=json.loads((OLD/'protocol.json').read_text())
    tasks=[r for r in protocol['tasks'] if r['arm']=='HARD_MGDA']
    bindings={}
    for r in tasks:
        r['stop_file']=str(OLD/r['task']/'stopped_complete.pt')
        r['evaluation_file']=str(OLD/r['task']/'evaluation_inputs.pt')
        for key in ('checkpoint','data','history','stop_file','evaluation_file'):
            bindings[r[key]]=sha(r[key])
    for directory in (HERE,ROOT/'research_branches/hard_clean_pilot_v2_20261110'):
        for path in directory.glob('*.py'):bindings[str(path)]=sha(path)
    old_bindings=json.loads((OLD/'bindings.json').read_text())
    for path,h in old_bindings.items():
        if path.endswith('.py'):
            assert sha(path)==h,'historical source changed '+path
            bindings[path]=h
    save(OUT/'protocol.json',dict(tasks=tasks,bindings=bindings,
        objective='same five reference-relative hard risks; only combiner and stable LOO arithmetic change',
        train_draws=16,evaluation_draws=32,times=protocol['evaluation_times'],
        updates_each=1,raw_only=True,evaluation_inputs='old development inputs, not independent confirmation',
        GPU_seconds_limit=540,wall_minutes_limit=15,no_long_training=True,no_reverse_generation=True,
        estimator='exact absent triangle squared count; remaining groups LOO',
        combiner='project fixed arithmetic mean onto estimated common non-ascent cone; no objective omitted',
        zero_feedback='unidentified under this batch, never certified safe',
        optimizer_semantics='original AdamW state and weight decay; gradient projection does not constrain actual parameter displacement'))
    print('prepared four conditions; no GPU/model calls')


def main():
    start=time.time();summary=dict(start_epoch=start,forwards=0,backwards=0,optimizer_updates=0,conditions=[])
    def expire(*_):raise TimeoutError('540-second allocation limit')
    signal.signal(signal.SIGALRM,expire);signal.alarm(535)
    signal.signal(signal.SIGTERM,expire)
    protocol=json.loads((OUT/'protocol.json').read_text())
    save(OUT/'allocation.json',dict(start_epoch=start,hard_stop_epoch=start+540,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid()))
    try:
        setup();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert '2080 Ti' in torch.cuda.get_device_name(0)
        save(OUT/'environment.json',g.common.environment('cuda'))
        for path,h in protocol['bindings'].items():assert sha(path)==h,'bound file changed '+path
        for row in protocol['tasks']:
            if time.time()-start>420:raise TimeoutError('insufficient reserved closeout time')
            folder=OUT/row['task'];folder.mkdir(exist_ok=False);tick=time.time()
            original,cfg,case,m,o,e,rng,b=load(row)
            saved=torch.load(row['stop_file'],map_location='cpu')
            state=saved['state'];b=restore(state,m,o,e,rng);u=b.completed+1
            params=list(m.parameters());names=[n for n,_ in m.named_parameters()]
            aux=np.random.default_rng();aux.bit_generator.state=copy.deepcopy(saved['auxiliary_rng'])
            fixed=torch.load(row['evaluation_file'],map_location='cpu')['inputs']
            order=np.argsort(fixed[0]['ids'])
            core=HardScore(case['space'].types,fixed[0]['query'][:,order],fixed[0]['f'][order],
                           fixed[0]['labels'][order],case['n_families'])
            canonical=fixed[0]['ids'][order];analytic=ExactAbsentTriangles(core)

            def evaluate(label):
                boundary=g.digest(b.checkpoint(state['binding']));aux_hash=g.digest(aux.bit_generator.state)
                rows=[]
                with b.validation():
                    m.eval()
                    for j,inp in enumerate(fixed):
                        token=('zero_safe_eval',label,j)
                        ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                        try:
                            with torch.no_grad():
                                graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                                g.common.check_prob_device(p,inp['f'],case,m.prior)
                                terms=supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                            summary['forwards']+=1
                            ix=np.argsort(inp['ids']);prob=p.cpu().double().numpy()[ix]
                            er=np.random.default_rng(row['seed']+810001+100+j)
                            bits=(er.random((32,len(prob)))<prob).astype(np.uint8)
                            costs=core.batch(bits);y=inp['labels'];relations=[]
                            for f in range(case['n_families']):
                                pos=(inp['f']==f)&(y==1);neg=(inp['f']==f)&(y==0)
                                relations.append(dict(family=f,positive_n=int(pos.sum()),negative_n=int(neg.sum()),
                                    positive_CE=float(terms[pos].mean()) if pos.any() else None,
                                    negative_CE=float(terms[neg].mean()) if neg.any() else None,
                                    expected_edges=float(prob[core.f==f].sum()),reference_edges=int(core.ref['counts'][f])))
                            rows.append(dict(readout='raw',t=inp['t'],input_sha256=input_hash([inp]),
                                hard_cost_mean=costs.mean(0).tolist(),hard_cost_std=costs.std(0,ddof=1).tolist(),
                                positive_CE=float(terms[y==1].mean()),negative_CE=float(terms[y==0].mean()),
                                CE=float(terms.sum()/case['space'].total),relations=relations))
                            np.savez_compressed(folder/f'{label}_t{inp["t"]}.npz',probability=prob,bits=bits,costs=costs,ids=canonical)
                        finally:ctx.close()
                assert g.digest(b.checkpoint(state['binding']))==boundary,'evaluation state pollution'
                assert g.digest(aux.bit_generator.state)==aux_hash
                save(folder/f'{label}.json',rows)
                return rows

            before=evaluate('before')
            history=[json.loads(s) for s in Path(row['history']).read_text().splitlines()]
            before_parameters=torch.cat([p.detach().flatten() for p in params]).clone()
            with b.step(u):
                inp=g.one_input(case,cfg,u,rng);ih=input_hash([inp])
                assert ih==history[u-1]['input_sha256']
                mainrng=g.digest(g.rng_capture(rng,'cuda'));m.train();o.zero_grad(set_to_none=True)
                token=('zero_safe_update',u)
                ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                try:
                    graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                    summary['forwards']+=1;g.common.check_prob_device(p,inp['f'],case,m.prior)
                    ix=np.argsort(inp['ids']);assert np.array_equal(inp['ids'][ix],canonical)
                    assert np.array_equal(inp['labels'][ix],core.reference)
                    prob=p.detach().cpu().double().numpy()[ix]
                    margin=(logits[:,1]-logits[:,0])[torch.as_tensor(ix,device='cuda')]
                    free=(prob>0)&(prob<1)
                    assert np.max(abs(torch.sigmoid(margin.detach().double()).cpu().numpy()[free]-prob[free]))<2e-6
                    bits=(aux.random((16,len(prob)))<prob).astype(np.uint8)
                    costs=core.batch(bits);all_dg=loo_gradient(prob,bits,costs)
                    av,ag=analytic.value_gradient(prob);all_dg[3]=ag;dg=all_dg[core.active];dg[:,~free]=0
                    used=None;vectors=[]
                    for j,v in enumerate(dg):
                        grads=torch.autograd.grad(margin,params,grad_outputs=torch.as_tensor(v,device='cuda',dtype=margin.dtype),
                            retain_graph=j<len(dg)-1,allow_unused=True)
                        support=[x is not None for x in grads]
                        if used is None:used=support
                        assert used==support
                        vectors.append(torch.cat([(torch.zeros_like(p0) if x is None else x).detach().flatten() for p0,x in zip(params,grads)]))
                        summary['backwards']+=1
                    G=torch.stack(vectors);assert torch.isfinite(G).all()
                    H=(G.double()@G.double().T).cpu().numpy()
                    coeff,meta=project_mean(H);old,_=solve_gram(H)
                    combined=torch.as_tensor(coeff,device='cuda',dtype=G.dtype)@G
                    old_vector=torch.as_tensor(old,device='cuda',dtype=G.dtype)@G
                    actual_inner=(G.double()@combined.double()).cpu().numpy()
                    norm=float(combined.double().norm());assert norm>1e-12,'no usable direction, do not force update'
                    # Check float32 direction separately; this is precision, not a changed loss tolerance.
                    roundoff=64*np.finfo(np.float32).eps*np.sqrt(np.maximum(np.diag(H),0))*norm
                    assert np.all(actual_inner>=-roundoff),'float32 direction violates estimated cone beyond roundoff'
                    offset=0
                    for p0,use in zip(params,used):
                        p0.grad=combined[offset:offset+p0.numel()].view_as(p0).clone() if use else None
                        offset+=p0.numel()
                    assert g.digest(g.rng_capture(rng,'cuda'))==mainrng
                    b.mark_optimizer_started();o.step();e.update(m,u);e.at_update_boundary=True
                    summary['optimizer_updates']+=1
                    displacement=torch.cat([p.detach().flatten() for p in params])-before_parameters
                    actual_adam=(G.double()@displacement.double()).cpu().numpy()
                    assert torch.isfinite(displacement).all()
                    groups={};offset=0
                    for name,p0 in zip(names,params):
                        d=displacement[offset:offset+p0.numel()];offset+=p0.numel()
                        group=name.split('.')[0];groups[group]=groups.get(group,0.)+float(d.double().square().sum())
                    failure=json.loads(next((OLD/row['task']).glob('zero_gradient_u*.json')).read_text())
                    info=dict(task=row['task'],update=u,t=inp['t'],input_sha256=ih,main_rng_sha256=mainrng,
                        old_cost_max_difference=float(np.max(abs(costs-np.array(failure['costs'])))),
                        cost_mean=costs.mean(0).tolist(),constant_cost_groups=np.flatnonzero(np.ptp(costs,axis=0)==0).tolist(),
                        exact_absent_risk=av,Gram=H.tolist(),coefficients=coeff.tolist(),solver=meta,
                        old_MGDA_norm=float(old_vector.double().norm()),new_gradient_norm=norm,
                        individual_norms=np.sqrt(np.maximum(np.diag(H),0)).tolist(),
                        float32_direction_inner_products=actual_inner.tolist(),roundoff_bounds=roundoff.tolist(),
                        actual_AdamW_norm=float(displacement.double().norm()),
                        estimated_loss_first_order_actual_AdamW=actual_adam.tolist(),
                        actual_AdamW_module_norms={k:v**.5 for k,v in groups.items()},
                        caveat='negative gradient passed to AdamW is not its actual displacement; no all-objective loss guarantee')
                    np.savez_compressed(folder/'failed_batch_replay.npz',probability=prob,bits=bits,costs=costs,ids=canonical,Gram=H)
                finally:ctx.close()
            assert b.completed==e.updates==u
            complete=b.checkpoint(state['binding'])
            checkpoint(folder/'temporary_complete.pt',dict(state=complete,auxiliary_rng=copy.deepcopy(aux.bit_generator.state),
                diagnostic_only=True,task=row,combiner='project_mean',parent_stop_sha256=sha(row['stop_file'])))
            # Exact saved state identity, next main input and auxiliary draw; no repeat optimization.
            b=restore(torch.load(folder/'temporary_complete.pt',map_location='cpu')['state'],m,o,e,rng)
            info['save_reload_boundary_exact']=True
            after=evaluate('after')
            info['mean_cost_before']=np.mean([r['hard_cost_mean'] for r in before],axis=0).tolist()
            info['mean_cost_after']=np.mean([r['hard_cost_mean'] for r in after],axis=0).tolist()
            info['seconds']=time.time()-tick
            save(folder/'result.json',info);summary['conditions'].append(info)
            save(OUT/'progress.json',summary)
            print(json.dumps({k:info[k] for k in ('task','update','old_MGDA_norm','new_gradient_norm','seconds')}),flush=True)
            del G,combined,old_vector,displacement,before_parameters,params,m,o,e,b,complete,state,saved
            torch.cuda.empty_cache()
        for path,h in protocol['bindings'].items():assert sha(path)==h,'input/source modified '+path
        summary['status']='complete';summary['bound_files_unchanged']=len(protocol['bindings'])
    except BaseException as exc:
        summary.update(status='stopped',error=repr(exc),traceback=traceback.format_exc())
    finally:
        summary.update(end_epoch=time.time(),elapsed_seconds=time.time()-start,
            GPU_hours_process=(time.time()-start)/3600,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0,
            full_generation_chains=0,formal_training=False)
        save(OUT/'receipt.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='conditions'}),flush=True)
    return 0 if summary['status']=='complete' else 2

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:raise SystemExit(main())
