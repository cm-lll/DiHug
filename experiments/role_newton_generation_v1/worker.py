import argparse,traceback,resource,signal
from generation_common import *

def run(arm,seed):
    tick=time.perf_counter();cpu=time.process_time();folder=OUT/f'{arm}_{seed}';folder.mkdir();status='failed';reason=None;done=0;reader=None
    def stop(s,f):(OUT/'STOP').touch()
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    try:
        check();f=frozen();records,targets=c.load_data();case=c.metadata_case(c.base,c.PANEL,targets)
        model,binding=old.get_model(seed,case);save(folder/'binding.json',binding)
        # Replicate original fixed validation on exact path before changing any reader.
        v=old.validate_device(model,c.base.fixed_validation(records,c.settings()),c.settings(),targets,check)
        ref=read(c.RESULT/f'jobs/broad_{seed}/validation_u006000.json');maximum=0.
        for x,y in zip(v['rows'],ref['rows']):
            assert (x['id'],x['condition'],x['t'])==(y['id'],y['condition'],y['t'])
            for k in ('ce','ce_sum_by_family_label','expected_edges'):maximum=max(maximum,float(abs(np.asarray(x[k])-np.asarray(y[k])).max()))
        assert len(v['rows'])==len(ref['rows']) and maximum<=2e-8
        save(folder/'validation.json',dict(maximum_difference=maximum,validation=v))
        reader=Reader(arm,folder);reader.install(model)
        for sample_seed in f['sampling_seeds']:
            for name in f['targets']:
                check();start=time.perf_counter();stem=f'{name}_{sample_seed}'
                reader.rows=[];reader.arrays={};states=[];diagnostics=[];reader.begin(64);reader.values=[]
                def commit(state,steps,rng):
                    assert reader.typ==4 and len(reader.current)==4 and len(reader.values)==4
                    diagnostics.append(reader.current);states.append(state.copy())
                    if len(steps)%8==0:
                        np.savez_compressed(folder/'partial.npz',state=state)
                        save(folder/'partial.json',dict(condition=name,sampling_seed=sample_seed,steps=steps,rng=rng,readouts=diagnostics))
                        save(folder/'progress.json',dict(completed_chains=done,condition=name,sampling_seed=sample_seed,steps=len(steps),elapsed_seconds=time.perf_counter()-tick))
                    reader.begin(steps[-1]['t']-1);reader.values=[]
                state,steps,initial=c.original_chain(model,case,targets[name],sample_seed,c.base,check,commit)
                assert len(reader.rows)==256 and len(states)==64
                edges,fam=c.base.c.state_edges(case,state);graphfile=folder/(stem+'.npz')
                np.savez_compressed(graphfile,types=case['space'].types,edges=edges.T,relations=fam+1,schema=np.asarray(case['space'].pairs))
                trajectory=folder/(stem+'_trajectory.npz')
                np.savez_compressed(trajectory,states=np.asarray(states),**{k:np.asarray(v) for k,v in reader.arrays.items()})
                save(folder/(stem+'_steps.json'),steps);save(folder/(stem+'_reads.json'),diagnostics)
                metrics=c.graph(case,edges,c.NAMES,check)
                result=dict(arm=arm,seed=seed,condition=name,sampling_seed=sample_seed,checkpoint_update=6000,weight_sha256=binding['weight']['sha256'],
                    target_fingerprint=targets[name].fingerprint,initial_state_sha256=initial,graph_file=str(graphfile.relative_to(OUT)),graph_sha256=sha(graphfile),
                    steps_file=str((folder/(stem+'_steps.json')).relative_to(OUT)),steps_sha256=sha(folder/(stem+'_steps.json')),
                    trajectory_file=str(trajectory.relative_to(OUT)),trajectory_sha256=sha(trajectory),reads_file=str((folder/(stem+'_reads.json')).relative_to(OUT)),reads_sha256=sha(folder/(stem+'_reads.json')),
                    metrics=metrics,target_errors=c.error_stats(metrics,targets[name].as_dict()),seconds=time.perf_counter()-start,newly_executed_chain=True,
                    solve_seconds=sum(r['total_checked_seconds'] for r in reader.rows),original_contract_reads=sum(r.get('original_contract_pass',False) for r in reader.rows),bounded_tail_reads=sum(r['status']=='converged_bounded_tail' for r in reader.rows))
                save(folder/(stem+'_result.json'),result);done+=1
                print(arm,seed,done,'/88',name,sample_seed,'seconds',round(result['seconds'],3),flush=True)
        assert done==88;status='completed'
    except BaseException as exc:
        reason=repr(exc);traceback.print_exc();(OUT/'STOP').touch()
        if reader is not None:save(folder/'interrupted_reads.json',reader.rows)
    finally:
        if reader is not None:reader.close()
        save(folder/'receipt.json',dict(status=status,reason=reason,arm=arm,seed=seed,completed_chains=done,wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,training=0,GPU=0))
    return int(status!='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args();sys.exit(run(a.arm,a.seed))
