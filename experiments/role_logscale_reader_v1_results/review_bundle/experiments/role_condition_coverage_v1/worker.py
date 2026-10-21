"""CPU probe/train/sample jobs. Scientific weights always start from original initialization."""
import argparse,resource,signal,traceback,copy,gc
from collections import Counter
from common import *
from transaction import UpdateBoundary

def progress(folder,phase,**kw):
    r=dict(phase=phase,utc_epoch=time.time(),elapsed_from_activation=time.time()-read(RESULT/'activation.json')['activation_epoch'],**kw)
    save(folder/'progress.json',r);print(json.dumps(r),flush=True)

def validate(model,records,targets,cfg,folder,u):
    before=base.c.rng_save(np.random.default_rng(0),'cpu')
    values=base.fixed_validation(records,cfg)
    v=validate_device(model,values,cfg,targets,check)
    after=base.c.rng_save(np.random.default_rng(0),'cpu')
    assert before['python']==after['python']
    assert torch.equal(before['torch'],after['torch'])
    assert before['numpy'][0]==after['numpy'][0] and np.array_equal(before['numpy'][1],after['numpy'][1]) and before['numpy'][2:]==after['numpy'][2:]
    save(folder/f'validation_u{u:06d}.json',v)
    return v

def checkpoint(folder,model,optimizer,u,binding,tag):
    assert int(optimizer.state.get(model.backbone.lin_out_E.weight,{}).get('step',0))==u
    record=dict(model=base.c.cpu_tree(model.state_dict()),optimizer=base.c.cpu_tree(optimizer.state_dict()),
        completed_updates=u,binding=binding,budget=read(RESULT/'activation.json'),
        random=base.c.rng_save(np.random.default_rng(0),'cpu'),
        training_inputs='counter-based(seed,update,micro) and frozen configuration; nextupdate=completed_updates',
        next_update=u,trained_continuously=True)
    path=folder/(tag+'.pt');assert not path.exists()
    h=base.atomic_weights(path,record)
    append(folder/'checkpoint_index.jsonl',dict(path=str(path.relative_to(RESULT)),sha256=h,completed_updates=u,tag=tag))
    return dict(path=str(path.relative_to(RESULT)),sha256=h,completed_updates=u)

def probe(a,records,targets,cfg):
    folder=a.folder;case=records[0]['case'];m,init=make_model(case,cfg,a.seed);o=base.opt(m,cfg)
    names=read(RESULT/'conditions.json')['training'][a.arm];costs=[];sweeps=[]
    for name in names:
        cc=next(r['case'] for r in records if r['arm']==a.arm and r['condition']==name and r['split']=='train')
        times=[];m.set_target(targets[name])
        for j,t in enumerate((1,32,64)):
            check();inp=[base.c.train_input(cc,64,4,a.seed,100*j+k,np.random.default_rng([2026102126,a.seed,j,k]),force_t=t) for k in (0,1)]
            r=base.double_update(m,o,cc,inp,64,check=check);times.append(r['seconds'])
        costs.append(dict(condition=name,role_kinds=len(targets[name].roles[3]),seconds=times,
             first_seconds=times[0],stable_P90=float(np.quantile(times[1:],.9))))
    m.eval();dense=np.repeat(case['active'].astype(np.uint8),case['space'].counts)
    kernel=base.MarginalKernel(case['prior'],base.c.schedule(64))
    for name in read(RESULT/'conditions.json')['final_targets']:
        m.set_target(targets[name]);dur=[]
        for j,t in enumerate((1,32,64,32)):
            check();start=time.monotonic()
            with torch.no_grad():base.reverse_step(m,case,dense,kernel,t,4,a.seed,np.random.default_rng([2026102127,j]),'cpu',check_time=check)
            dur.append(time.monotonic()-start)
        sweeps.append(dict(condition=name,role_kinds=len(targets[name].roles[3]),complete4block_seconds=dur,maximum=max(dur)))
    v=validate_device(m,base.fixed_validation(records,cfg),cfg,targets,check)
    # Disposable save/strict reload cost; never used as formal initial state.
    path=folder/'engineering_roundtrip.pt';start=time.monotonic()
    h=base.atomic_weights(path,dict(model=m.state_dict(),optimizer=o.state_dict(),engineering_only=True))
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,ck['model'][k]) for k,v in m.state_dict().items())
    save_cost=time.monotonic()-start
    save(folder/'probe.json',dict(status='completed',arm=a.arm,seed=a.seed,updates=costs,sweeps=sweeps,
        validation_seconds=v['seconds'],save_reload_seconds=save_cost,checkpoint_bytes=path.stat().st_size,
        training_worst_P90=max(x['stable_P90'] for x in costs),sweep_worst_seconds=max(x['maximum'] for x in sweeps),
        parameters=init['parameters'],precision='unchangedFP32/float64',disposable=True,
        engineering_updates=3*len(names),scientific_updates=0,science_chains=0))
    # Retain fixture weights only locally and label explicitly, never imported by training.

def train(a,records,targets,cfg):
    gate=read(RESULT/'cost_gate.json');auth=read(RESULT/'learning_authorization.json')
    assert gate['passed'] and auth['cost_gate_sha256']==sha(RESULT/'cost_gate.json')
    assert auth['activation_sha256']==sha(RESULT/'activation.json')
    folder=a.folder;binding=check_binding();case=records[0]['case'];model,init=make_model(case,cfg,a.seed)
    optimizer=base.opt(model,cfg);assert not optimizer.state
    initial_path=folder/'initial.pt';initial_hash=base.atomic_weights(initial_path,model.state_dict())
    save(folder/'initialization.json',dict(**init,initial_sha256=initial_hash,arm=a.arm,seed=a.seed,
        paired_source_hash=init['source_sha256'],optimizer_fresh=True,engineering_weights_imported=False))
    complete=0;counts=Counter();boundary=UpdateBoundary(RESULT,read(RESULT/'activation.json')['deadline_epoch'],reserve=30)
    last_valid=True
    try:
        for u in range(6001):
            boundary.check()
            if u in cfg['checkpoint_updates']:checkpoint(folder,model,optimizer,u,binding,f'checkpoint_u{u:06d}')
            if u in cfg['validation_updates']:
                progress(folder,'validation',completed_updates=u)
                validate(model,records,targets,cfg,folder,u)
            if u==6000:break
            name,cc,inp,receipt=inputs_for(records,targets,a.arm,a.seed,u,cfg)
            with boundary.pair():
                model.set_target(targets[name])
                result=base.double_update(model,optimizer,cc,inp,64,check=boundary.check)
                complete=u+1;counts[name]+=1
                assert int(optimizer.state[model.backbone.lin_out_E.weight]['step'])==complete
                append(folder/'updates.jsonl',dict(seed=a.seed,arm=a.arm,update=complete,condition=name,
                    **receipt,**result))
            if complete%100==0:
                save(folder/'condition_counts.json',dict(updates=dict(counts),samples={k:2*v for k,v in counts.items()}))
                progress(folder,'training',completed_updates=complete,last_loss=result['loss'],batch_seconds=result['seconds'],condition_updates=dict(counts))
        check();final_path=folder/'final6000.pt';h=base.atomic_weights(final_path,model.state_dict())
        save(folder/'final_weight.json',dict(arm=a.arm,seed=a.seed,completed_updates=complete,path=str(final_path.relative_to(RESULT)),sha256=h,
             source_initialization=init,selection='final6000 only; validation/test never choose checkpoint'))
        save(folder/'condition_counts.json',dict(updates=dict(counts),samples={k:2*v for k,v in counts.items()}))
        assert sum(counts.values())==6000
        progress(folder,'training_completed',completed_updates=complete)
    except BaseException:
        last_valid=all(torch.isfinite(p).all() for p in model.parameters())
        if last_valid:
            checkpoint(folder,model,optimizer,complete,binding,f'emergency_u{complete:06d}')
        else:save(folder/'invalid_state.json',dict(completed_updates=complete,nonfinite_parameters=True,last_milestones_retained=True))
        raise

def sample(a,records,targets,cfg):
    folder=a.folder;check_binding();sealed=read(RESULT/'sealed_final_weights.json')
    assert sealed['all_six_complete'] and len(sealed['weights'])==6
    for w in sealed['weights']:assert sha(RESULT/w['path'])==w['sha256']
    own=next(w for w in sealed['weights'] if w['arm']==a.arm and w['seed']==a.seed)
    # Metadata-only inference case, no training edges/identities.
    case=metadata_case(base,PANEL,targets)
    model,_=make_model(case,cfg,a.seed)
    model.load_state_dict(torch.load(RESULT/own['path'],map_location='cpu',weights_only=True),strict=True)
    model.eval();gen=folder/'generated';gen.mkdir();done=0
    for sampling_seed in cfg['sampling_seeds']:
        for name in read(RESULT/'conditions.json')['final_targets']:
            check();tick=time.monotonic();stem=f'{name}_{sampling_seed}'
            def commit(state,steps,rng_state):
                if len(steps)%8==0:
                    dummy=np.random.default_rng();dummy.bit_generator.state=rng_state
                    base.atomic_weights(folder/'sampling_partial.pt',dict(state=state,steps=steps,RNG=base.c.rng_save(dummy,'cpu'),
                         target=name,sampling_seed=sampling_seed,weight=own,completed_step=steps[-1]['t'],budget=read(RESULT/'activation.json')))
                    progress(folder,'sampling',completed_chains=done,condition=name,sampling_seed=sampling_seed,completed_steps=len(steps))
            state,steps,initial=original_chain(model,case,targets[name],sampling_seed,base,check,commit)
            edges,fam=base.c.state_edges(case,state);path=gen/(stem+'.npz')
            np.savez_compressed(path,types=case['space'].types,edges=edges.T,relations=fam+1,schema=np.asarray(case['space'].pairs))
            save(gen/(stem+'_steps.json'),steps)
            metrics=graph(case,edges,NAMES,check)
            r=dict(arm=a.arm,seed=a.seed,condition=name,sampling_seed=sampling_seed,checkpoint_update=6000,
                graph_file=str(path.relative_to(RESULT)),graph_sha256=sha(path),steps_file=str((gen/(stem+'_steps.json')).relative_to(RESULT)),
                steps_sha256=sha(gen/(stem+'_steps.json')),target_fingerprint=targets[name].fingerprint,
                weight_sha256=own['sha256'],initial_state_sha256=initial,metrics=metrics,target_errors=error_stats(metrics,targets[name].as_dict()),
                seconds=time.monotonic()-tick,newly_executed_chain=True)
            save(gen/(stem+'_result.json'),r);done+=1
    assert done==88;check();progress(folder,'sampling_completed',completed_chains=done)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('probe','train','sample'))
    p.add_argument('--arm',choices=('narrow','broad'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--folder',type=Path,required=True);a=p.parse_args();a.folder.mkdir(exist_ok=a.mode=='sample',parents=True)
    start=time.monotonic();cpu=time.process_time();status='failed';reason=None
    def stop(sig,frame):(RESULT/'STOP').touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    try:
        check();check_binding();records,targets=load_data();cfg=settings()
        globals()[a.mode](a,records,targets,cfg);status='completed'
    except BaseException as exc:
        reason=repr(exc);traceback.print_exc();status='stopped' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed'
    finally:
        save(a.folder/(a.mode+'_receipt.json'),dict(status=status,reason=reason,mode=a.mode,arm=a.arm,seed=a.seed,
            wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,device='cpu',GPU_used=False))
    return 0 if status=='completed' else 1
if __name__=='__main__':sys.exit(main())
