"""Fixed paired CPU experiment; cost gate precedes ALL scientific learning."""
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from hsgd.bw_preflight import common as c
from hsgd.runtime import predict,reverse_step
from hsgd.kernel import MarginalKernel
from hsgd.losses import supported_binary_ce
from .model import paired_models,extract_target,sha
from .panel import make_panel,save_panel,augment,NAMES,nx_graph
from .metrics import graph,compare

def save(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def emit(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    print(json.dumps(obj,allow_nan=False),flush=True)

def opt(model,cfg):
    kwargs=dict(cfg['optimizer']);assert kwargs.pop('type')=='AdamW';kwargs['betas']=tuple(kwargs['betas'])
    return torch.optim.AdamW(model.parameters(),**kwargs)

def set_condition(model,condition,targets):model.set_target(targets[condition] if model.mode=='C' else None)

def fixed_validation(records,cfg):
    values=[]
    for i,rec in enumerate([r for r in records if r['split']=='validation']):
        for j,t in enumerate(cfg['validation_times']):
            rng=np.random.default_rng([cfg['validation_seed'],i,j])
            case=augment(rec['case'],rng)
            inp=c.train_input(case,cfg['T'],cfg['rounds'],cfg['validation_seed'],i*3+j,rng,force_t=t)
            values.append((rec['id'],rec['condition'],case,inp,i*3+j))
    return values

def validate(model,values,cfg,targets,check):
    import random
    global_state=(random.getstate(),np.random.get_state(),torch.get_rng_state());training=model.training
    model.eval();rows=[];started=time.perf_counter()
    try:
        with torch.no_grad():
            for identifier,condition,case,inp,index in values:
                check();set_condition(model,condition,targets);plan=c.plan_for(case,[cfg['validation_seed'],index,21],cfg['rounds'])
                ce=np.zeros((case['n_families'],2));counts=np.zeros_like(ce,dtype=int);expected=np.zeros(case['n_families'])
                for k in range(cfg['rounds']):
                    check();ids,f,q=plan.block(k)
                    logits,p=predict(model,case['space'].types,inp['edges'],inp['edge_families'],q,f,inp['t']/cfg['T'],details=True)
                    y=np.isin(ids,case['clean_ids']).astype(np.float32)
                    item=supported_binary_ce(logits,torch.from_numpy(y),model.prior[torch.from_numpy(f)]).numpy()
                    prob=c.check_prob(p,f,case);np.add.at(expected,f,prob)
                    for rel in range(case['n_families']):
                        for label in (0,1):
                            mask=(f==rel)&(y==label);counts[rel,label]+=int(mask.sum());ce[rel,label]+=float(item[mask].sum())
                rows.append(dict(id=identifier,condition=condition,t=inp['t'],ce=float(ce.sum()/case['space'].total),
                    ce_sum_by_family_label=ce.tolist(),counts_by_family_label=counts.tolist(),expected_edges=expected.tolist(),
                    expected_edge_bias=(expected-case['counts']).tolist(),normalization='full schema S; inactive zero slots have loss0'))
    finally:
        random.setstate(global_state[0]);np.random.set_state(global_state[1]);torch.set_rng_state(global_state[2]);model.train(training)
    return dict(seconds=time.perf_counter()-started,rows=rows,mean_ce=float(np.mean([r['ce'] for r in rows])))

def main(out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());save(out/'resolved_config.json',cfg)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(out/'environment.json',c.environment('cpu'))
    started=time.monotonic();cpu_start=time.process_time();stage='preparation';status='incomplete';reason=None
    def check():
        if (out/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()>deadline-10:raise TimeoutError('CPU budget reached; no extension')
    try:
        records,targets=make_panel(cfg['panel_seed']);panel_hash=save_panel(records,targets,out/'panel.json')
        case=records[0]['case'];values=fixed_validation(records,cfg);train=[r for r in records if r['split']=='train']
        probe_case=next(r['case'] for r in records if r['condition']=='extreme')
        # Predeclared cost probe uses full original widths/layers, disposable paired models.
        stage='cost_probe';models=paired_models(case,cfg,2026102099);probe={};rng=np.random.default_rng(2026102099)
        setup_seconds=time.monotonic()-started
        for arm,model in models.items():
            check();set_condition(model,'extreme',targets);optimizer=opt(model,cfg);durations=[]
            for u in range(cfg['cost_probe_training_updates_per_arm']):
                check();st=time.perf_counter();inp=c.train_input(probe_case,cfg['T'],cfg['rounds'],2026102099,u,rng)
                c.update(model,optimizer,probe_case,inp,cfg['T'],'cpu');durations.append(time.perf_counter()-st)
            model.eval();inp=c.train_input(case,64,4,2026102099,41,rng,force_t=32);forward=[]
            with torch.no_grad():
                for _ in range(cfg['cost_probe_forward_repetitions']):
                    check();st=time.perf_counter();predict(model,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],.5)
                    forward.append(time.perf_counter()-st)
                dense=np.repeat(case['active'].astype(np.uint8),case['space'].counts);dense_step=[]
                kernel=MarginalKernel(case['prior'],c.schedule(64))
                for k in range(cfg['cost_probe_forward_repetitions']):
                    check();st=time.perf_counter()
                    reverse_step(model,case,dense,kernel,32,4,2026102099,rng,'cpu',check_time=check)
                    dense_step.append(time.perf_counter()-st)
            vt=validate(model,values,cfg,targets,check)['seconds']
            probe[arm]=dict(update_seconds=durations,stable_update_median=float(np.median(durations[3:])),
                stable_update_p90=float(np.quantile(durations[3:],.9)),forward_seconds=forward,
                forward_p90=float(np.quantile(forward[2:],.9)),validation_seconds=vt,
                dense_complete_step_seconds=dense_step,dense_step_p90=float(np.quantile(dense_step[2:],.9)),
                parameters=sum(p.numel() for p in model.parameters()),branch_calls=model.branch_calls,
                branch_timings=model.branch_timings)
            emit(out/'progress.jsonl',dict(phase='cost_probe',arm=arm,update_median=probe[arm]['stable_update_median'],forward_p90=probe[arm]['forward_p90']))
            del optimizer
        del models;import gc;gc.collect()
        # 24 A chains,24 B chains,48 C chains. 64*4 neural calls each.
        training_est=3*1000*sum(x['stable_update_p90'] for x in probe.values())
        sampling_est=sum(n*64*probe[a]['dense_step_p90'] for a,n in [('A',24),('B',24),('C',48)])
        validation_est=3*4*sum(x['validation_seconds'] for x in probe.values())
        forecast=cfg['cost_forecast_safety_factor']*(training_est+sampling_est+validation_est)+20
        remaining=deadline-time.monotonic();gate=forecast<remaining-cfg['exit_reserve_seconds']
        costs=dict(setup_seconds=setup_seconds,probe=probe,training_forecast=training_est,sampling_forecast=sampling_est,
            validation_forecast=validation_est,safety_factor=cfg['cost_forecast_safety_factor'],total_forecast=forecast,
            remaining_seconds=remaining,gate_passed=gate,panel_sha256=panel_hash,
            cost_scope='CPU single-thread full-model; sampling uses dense full-support Xt complete-step timings, not single-block timing; no quality used for gate')
        save(out/'costs.json',costs)
        if not gate:
            status='cost_blocked';reason='full frozen 3-seed panel does not fit budget; no scientific training launched';return 3
        results=[];initial=[];validation_records=[];training_records=[]
        stage='learning_and_sampling'
        for seed in cfg['train_seeds']:
            check();models=paired_models(case,cfg,seed);optimizers={a:opt(m,cfg) for a,m in models.items()}
            shared_names=list(models['A'].state_dict());degree_names=list(models['B'].branch.state_dict())
            initial.append(dict(seed=seed,shared_state_tensors=len(shared_names),degree_branch_tensors=len(degree_names),
                                shared_exactly_copied=True,degree_exactly_copied=True))
            rng=np.random.default_rng(seed);timings={a:[] for a in models};checkpoints=out/('seed'+str(seed));checkpoints.mkdir()
            for u in range(1001):
                if u in cfg['validation_updates']:
                    for arm,model in models.items():
                        check();v=validate(model,values,cfg,targets,check);entry=dict(seed=seed,arm=arm,update=u,**v)
                        validation_records.append(entry);save(checkpoints/('%s_validation_%04d.json'%(arm,u)),entry)
                        emit(out/'progress.jsonl',dict(phase='validation',seed=seed,arm=arm,update=u,mean_ce=v['mean_ce']))
                if u==1000:break
                check();rec=train[(u%2)*4+int(rng.integers(4))];current=augment(rec['case'],rng)
                inp=c.train_input(current,cfg['T'],cfg['rounds'],seed,u,rng)
                input_digest=sha(dict(graph=rec['id'],t=inp['t'],round=inp['round_id'],edges=inp['edges'].tolist(),query=inp['ids'].tolist(),labels=inp['labels'].tolist()))
                torch_rng=torch.get_rng_state();post=None
                for arm,model in models.items():
                    set_condition(model,rec['condition'],targets);torch.set_rng_state(torch_rng)
                    begin=time.perf_counter();r=c.update(model,optimizers[arm],current,inp,cfg['T'],'cpu');dt=time.perf_counter()-begin
                    if post is None:post=torch.get_rng_state()
                    elif not torch.equal(post,torch.get_rng_state()):raise AssertionError('arms consume different torch random input')
                    timings[arm].append(dt)
                    entry=dict(seed=seed,arm=arm,update=u+1,input_sha256=input_digest,condition=rec['condition'],graph_id=rec['id'],
                        t=inp['t'],round=inp['round_id'],query_count=len(inp['ids']),loss=r['loss'],gradient_norm=r['gradient_norm'],seconds=dt)
                    training_records.append(entry)
                    with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                if (u+1)%100==0:emit(out/'progress.jsonl',dict(phase='training',seed=seed,update=u+1,remaining=deadline-time.monotonic()))
            save(checkpoints/'paired_initialization.json',initial[-1]);sample_rows=[]
            for arm,model in models.items():
                torch.save(model.state_dict(),checkpoints/(arm+'_final_1000.pt'))
                model.eval();save(checkpoints/(arm+'_branch_cost.json'),dict(calls=model.branch_calls,times=model.branch_timings,
                    update_median=float(np.median(timings[arm][5:])),update_p90=float(np.quantile(timings[arm][5:],.9))))
                for sample_seed in cfg['sampling_seeds']:
                    # A/B produce one actual graph each; evaluated against two targets below.
                    for condition in (['balanced','extreme'] if arm=='C' else ['balanced']):
                        check();set_condition(model,condition,targets);sampling=np.random.default_rng(sample_seed)
                        state=(sampling.random(case['space'].total)<np.repeat(case['prior'],case['space'].counts)).astype(np.uint8)
                        initial_state=state.copy();st=time.perf_counter()
                        kernel=MarginalKernel(case['prior'],c.schedule(64))
                        for t in range(64,0,-1):
                            check()
                            with torch.no_grad():state,report=reverse_step(model,case,state,kernel,t,4,sample_seed,sampling,'cpu',check_time=check)
                        edges,fam=c.state_edges(case,state);raw_hash=sha(dict(types=case['space'].types.tolist(),edges=edges.T.tolist()))
                        stem='%s_%s_%d'%(arm,condition,sample_seed)
                        np.savez_compressed(checkpoints/(stem+'.npz'),types=case['space'].types,edges=edges.T,relations=fam+1,schema=np.asarray(case['space'].pairs))
                        metrics=graph(case,edges,NAMES,check)
                        for evaluated_condition in ([condition] if arm=='C' else ['balanced','extreme']):
                            reference=next(r['case'] for r in records if r['condition']==evaluated_condition)
                            ref=graph(reference,reference['clean'],NAMES,check);comparison=compare(metrics,ref)
                            mg=case['space'].pairs.index((2,3));g_stats=metrics['typed_degree']['Genre']['by_relation'][mg]
                            ghist=np.asarray(g_stats['histogram']);high=int(ghist[12:].sum())
                            signature=sha(dict(degree=metrics['typed_degree'],triangles=metrics['typed_triangles'],components=metrics['component_sizes']))
                            row=dict(seed=seed,arm=arm,condition=evaluated_condition,sampling_seed=sample_seed,raw_graph_sha256=raw_hash,
                                reused_output=arm!='C',initial_state_sha256=sha(initial_state.tolist()),seconds=time.perf_counter()-st,
                                metrics=metrics,comparison=comparison,G_high_ge12=high,G_max=g_stats['max'],G_zero=g_stats['zero'],
                                G_W1=comparison['typed_relation_degree_W1']['Genre'][mg],G_joint_TV=comparison['joint_relation_degree_TV']['Genre'],
                                topology_invariant=signature)
                            sample_rows.append(row);results.append(row)
                        emit(out/'progress.jsonl',dict(phase='sample',seed=seed,arm=arm,condition=condition,sampling_seed=sample_seed,edges=metrics['edges'],seconds=time.perf_counter()-st))
            # Nonzero learned C response on identical frozen validation Xt, NOT trajectory equivalence after inputs fork.
            response=[];model=models['C'];model.eval()
            with torch.no_grad():
                for identifier,cond,current,inp,index in values[:3]:
                    pp=[]
                    for target in ['balanced','extreme']:
                        set_condition(model,target,targets)
                        pp.append(predict(model,current['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/64).numpy())
                    response.append(dict(t=inp['t'],max_p0_difference=float(np.max(np.abs(pp[1]-pp[0]))),mean_p0_difference=float(np.mean(np.abs(pp[1]-pp[0])))))
            save(checkpoints/'actual_condition_response.json',response)
            save(checkpoints/'generation.json',sample_rows)
            del models,optimizers;gc.collect()
        save(out/'generation.json',results);save(out/'initialization.json',initial)
        summary=[]
        for seed in cfg['train_seeds']:
            for condition in ['balanced','extreme']:
                for arm in ['A','B','C']:
                    rows=[r for r in results if r['seed']==seed and r['condition']==condition and r['arm']==arm]
                    summary.append(dict(seed=seed,condition=condition,arm=arm,n=8,
                        G_W1=float(np.mean([r['G_W1'] for r in rows])),G_joint_TV=float(np.mean([r['G_joint_TV'] for r in rows])),
                        G_high_mean=float(np.mean([r['G_high_ge12'] for r in rows])),G_max_mean=float(np.mean([r['G_max'] for r in rows])),
                        edges_mean=float(np.mean([r['metrics']['edges'] for r in rows])),LCC_mean=float(np.mean([r['metrics']['lcc'] for r in rows])),
                        triangles_mean=float(np.mean([r['metrics']['triangles'] for r in rows])),
                        closure_mean=float(np.mean([r['metrics']['imdb_closure']['path_ratio'] or 0 for r in rows])),
                        provably_distinct_lower_bound=len({r['topology_invariant'] for r in rows})))
        save(out/'seed_summary.json',summary);status='completed';return 0
    except BaseException as exc:
        reason=repr(exc);status='failed' if not isinstance(exc,(TimeoutError,InterruptedError)) else 'stopped';raise
    finally:
        save(out/'summary.json',dict(status=status,reason=reason,stage=stage,wall_seconds=time.monotonic()-started,
            CPU_seconds=time.process_time()-cpu_start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            budget_seconds=cfg['budget_seconds'],computation_cap=deadline-started,device='CPU',large_graph_runs=0,
            scientific_training_started=stage=='learning_and_sampling',no_automatic_extension=True))

if __name__=='__main__':
    raise SystemExit(main(sys.argv[1],float(sys.argv[2])))
