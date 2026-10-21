"""Explicitly budgeted paired training and unchanged complete sampling, GPU only."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
import pair_core as p
from pair_model import Reader,paired_initial

def payload(binding,u,models,opts,rng,deadline):
    return {'binding':binding,'updates':u,'models':{m.arm:p.cpu(m.state_dict()) for m in models},
        'optimizers':{m.arm:p.cpu(o.state_dict()) for m,o in zip(models,opts)},'rng':p.rng_save(rng,'cuda'),
        'deadline_epoch':deadline,'ordinary_mode':True,'formal_native':True}

def chain(model,case,cfg,seed,out,binding,check):
    device=next(model.parameters()).device.type
    out.mkdir(exist_ok=True);ck=out/'chain_last.pt';rng=np.random.default_rng(seed);kernel=p.MarginalKernel(case['prior'],p.c.schedule(cfg['T']))
    chain_binding=dict(binding,arm=model.arm,sample_seed=seed,model_digest=p.digest(model.state_dict()))
    start=time.perf_counter();last_save=time.monotonic()
    if ck.exists():
        receipt=p.read(out/'chain_latest.json')
        if p.sha(ck)!=receipt['sha256']:raise ValueError('chain hash mismatch')
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        if saved['binding']!=chain_binding:raise ValueError('chain source mismatch')
        state=saved['state'];t=saved['next_t'];p.rng_restore(saved['rng'],rng,device)
    else:
        pi=np.repeat(case['prior'],case['space'].counts);state=(rng.random(len(pi))<pi).astype(np.uint8);t=cfg['T']
    def commit():
        x={'binding':chain_binding,'state':state.copy(),'next_t':t,'rng':p.rng_save(rng,device),'last_completed_t':t+1,'ordinary_resume_not_bitwise_guaranteed':True}
        tmp=ck.with_name(ck.name+'.tmp');torch.save(x,tmp);tmp.replace(ck);p.save(out/'chain_latest.json',{'sha256':p.sha(ck),'next_t':t})
    model.eval();reader=Reader(check,out/'failed_reads',max_history=8).install(model)
    try:
        with torch.no_grad():
            while t>0:
                check();new,info=p.reverse_step(model,case,state,kernel,t,cfg['rounds'],seed,rng,device,check_time=check,trace=False)
                # One atomic algorithmic commit, after ALL original query blocks.
                state=new;t-=1
                p.emit(out/'steps.jsonl',dict(info,next_t=t,role_reader=reader.summary()))
                if time.monotonic()-last_save>30 or t==0:commit();last_save=time.monotonic()
        commit();check()
        edges,f=p.c.state_edges(case,state);path=out/'graph.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['edges'],edges) or str(z['binding'].item())!=p.digest(chain_binding):raise ValueError('conflicting final graph')
        else:
            with (out/'graph.npz.tmp').open('wb') as h:np.savez_compressed(h,types=case['space'].types,edges=edges,families=f,schema=np.asarray(case['space'].pairs),binding=np.array(p.digest(chain_binding)))
            (out/'graph.npz.tmp').replace(path)
        check();metrics=p.graph(case,edges,cfg['data']['type_names'],check)
        p.save(out/'metrics.json',metrics);p.save(out/'summary.json',{'status':'completed','T':cfg['T'],'next_t':0,'graph_sha256':p.sha(path),'binding':chain_binding,
            'wall_seconds_this_invocation':time.perf_counter()-start,'reader_calls_this_invocation':reader.total_calls,'role_reader':reader.summary(),'role_guidance':False,'postprocessing':False})
        return metrics
    except BaseException:
        # reverse_step rolls RNG back for any partial step. state/t still refer
        # to the last complete step, including STOP in a query block.
        commit();raise
    finally:reader.close()

def forecast_check(preflight,cfg,budget):
    if not preflight.get('target_GPU_gate_passed') or not preflight.get('full_IMDB_cost_complete') or preflight.get('status')!='passed_target_GPU':raise ValueError('full GPU preflight required; fixtures/partial evidence refused')
    checks=preflight.get('checks')
    if not checks or len(checks)!=2:raise ValueError('missing numerical checks')
    estimate=preflight['cost']['forecast']['total_seconds_with_25percent_and_600s_margin']
    if estimate>budget*.8:raise ValueError(f'complete experiment estimate {estimate:.1f}s exceeds80% of confirmed budget {budget:.1f}s; no automatic reduction')
    return estimate

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',type=Path,required=True);a.add_argument('--deadline',type=float,required=True);a.add_argument('--resume',action='store_true');x=a.parse_args()
    out=x.output;cfg=p.config();status='failed';error=None;stage='initialize';completed=[];start=time.time()
    interrupted={'value':False}
    def sig(signum,frame):interrupted['value']=True
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,sig)
    def check():
        if interrupted['value'] or (out/'STOP').exists():raise InterruptedError('safe STOP')
        if time.time()>x.deadline-60:raise TimeoutError('confirmed total deadline; save exit reserve60s')
    try:
        package_hash=p.verify_package();p.setup('cuda',False);p.acquire_guard(out,'cuda');env=p.environment('cuda');pf=p.read(x.preflight/'preflight.json')
        if pf.get('package_manifest_sha256')!=package_hash or pf.get('config_sha256')!=p.sha(p.ROOT/'config.json') or pf.get('runtime_binding')!=p.bound_environment(env):raise ValueError('preflight source/config/runtime binding mismatch; no software/device bypass')
        activation=p.read(out/'activation.json')
        if activation['deadline_epoch']!=x.deadline:raise ValueError('budget was reset')
        remaining=x.deadline-time.time()
        if not x.resume:forecast_check(pf,cfg,remaining)
        case,target=p.dataset(cfg);binding={'package_manifest_sha256':package_hash,'config_sha256':p.sha(p.ROOT/'config.json'),'data_sha256':cfg['data']['sha256'],
            'target_sha256':target.fingerprint,'runtime':p.bound_environment(env),'deadline_epoch':x.deadline,'initialization':'fresh, exact common tensor copy; no old checkpoint imports'}
        if (out/'binding.json').exists() and p.read(out/'binding.json')!=binding:raise ValueError('existing run binding conflict; old provenance preserved')
        import shutil
        if shutil.disk_usage(out).free<cfg['minimum_free_disk_bytes']:raise OSError('less than16GiB free on actual persistent output volume')
        p.save(out/'binding.json',binding);p.save(out/'resolved_config.json',cfg);p.save(out/'environment.json',env);p.save(out/'anonymous_target.json',target.as_dict())
        p.save(out/'storage.json',{'output_path':str(out.resolve()),'disk_free_bytes':shutil.disk_usage(out).free,'persistent_storage_requires_operator_confirmation':True,
            'estimated_permanent_checkpoints_bytes':int(3*13*2*6658481*4*3),'recovery_files_maximum_per_seed':2,'full_state_and_RNG_included':True})
        p.save(out/'source_hashes.json',p.sources());p.save(out/'preflight_binding.json',{'path':str(x.preflight),'sha256':p.sha(x.preflight/'preflight.json')})
        reference=p.graph(case,case['clean'],cfg['data']['type_names'],check);p.save(out/'reference_metrics.json',reference)
        for seed in cfg['train_seeds']:
            check();folder=out/f'seed{seed}';folder.mkdir(exist_ok=True);checkpoints=folder/'checkpoints'
            pair_binding=dict(binding,seed=seed);models=paired_initial(case,cfg,seed);rng=np.random.default_rng(seed)
            torch.cuda.manual_seed(seed)
            for m in models:m.to('cuda');p.condition(m,target)
            opts=[p.optimizer(m,cfg) for m in models];u=0;reader=Reader(lambda:None,folder/'failed_reads',max_history=8).install(models[1]);safe_pair=True
            try:
                if checkpoints.exists() and list(checkpoints.glob('pair_u*.pt')):
                    if not x.resume:raise ValueError('fresh-start refuses existing training checkpoints')
                    ck=p.load_checkpoint(checkpoints,pair_binding);u=ck['updates']
                    for m,o in zip(models,opts):m.load_state_dict(ck['models'][m.arm],strict=True);o.load_state_dict(ck['optimizers'][m.arm])
                    p.rng_restore(ck['rng'],rng,'cuda')
                    if p.digest(p.rng_save(rng,'cuda'))!=p.digest(ck['rng']):raise ValueError('RNG restore contents mismatch')
                    for m,o in zip(models,opts):
                        if p.digest(m.state_dict())!=p.digest(ck['models'][m.arm]) or p.digest(o.state_dict())!=p.digest(ck['optimizers'][m.arm]):raise ValueError('model/optimizer restore contents mismatch')
                    p.emit(folder/'resume.jsonl',{'updates':u,'checkpoint_payload':ck['payload_digest'],'loaded_contents_exact':True,'future_ordinary_trajectory_not_guaranteed':True})
                else:
                    initial=payload(pair_binding,0,models,opts,rng,x.deadline);p.checkpoint(checkpoints,initial)
                    assert p.digest(initial['models']['base'])==p.digest(initial['models']['role'])
                    p.save(folder/'initialization.json',{'common_tensors_exact':True,'independent_optimizers':True,'initial_weights_digest':p.digest(initial['models']['base']),
                        'parameter_counts':{m.arm:sum(q.numel() for q in m.parameters()) for m in models}})
                if u==0 and not (folder/'validation_00000000.json').exists():
                    p.save(folder/'validation_00000000.json',{m.arm:p.validate(m,case,target,cfg,check) for m in models})
                last_save=time.monotonic();stage='training'
                while u<cfg['max_updates']:
                    check();safe_pair=False;pair_tick=time.perf_counter();inputs=p.input_pair(case,cfg,seed,u);rows={}
                    # STOP is honored at the paired optimizer boundary. Neither
                    # arm gets an extra committed update because of manual STOP.
                    for m,o in zip(models,opts):rows[m.arm]=p.update(m,o,case,inputs,cfg)[0]
                    u+=1;safe_pair=True
                    if any(a['input_sha256']!=b['input_sha256'] or a['query_sha256']!=b['query_sha256'] for a,b in zip(rows['base']['parts'],rows['role']['parts'])):raise AssertionError('unpaired inputs')
                    p.emit(folder/'training.jsonl',{'completed_updates':u,'seed':seed,'arms':rows,'pair_input_and_updates_seconds':time.perf_counter()-pair_tick,
                        'elapsed_seconds':time.time()-activation['activation_epoch'],'memory':p.c.memory('cuda'),'role_reader':reader.summary()})
                    p.save(out/'progress.json',{'phase':'training','seed':seed,'completed_paired_updates':u,'arm_updates_equal':True})
                    if u%cfg['checkpoint_every_updates']==0:
                        p.checkpoint(checkpoints,payload(pair_binding,u,models,opts,rng,x.deadline));last_save=time.monotonic()
                    elif time.monotonic()-last_save>=1800:
                        p.rolling_recovery(checkpoints,payload(pair_binding,u,models,opts,rng,x.deadline));last_save=time.monotonic()
                    if u%cfg['validation_every_updates']==0:
                        p.save(folder/f'validation_{u:08d}.json',{m.arm:p.validate(m,case,target,cfg,check) for m in models})
                p.checkpoint(checkpoints,payload(pair_binding,u,models,opts,rng,x.deadline))
            except (InterruptedError,TimeoutError):
                if safe_pair:p.checkpoint(checkpoints,payload(pair_binding,u,models,opts,rng,x.deadline))
                raise
            except BaseException as exc:
                p.save(folder/'training_failure.json',{'error':repr(exc),'last_completed_in_memory':u,'partial_pair':not safe_pair,'resume_only_latest_durable_valid_checkpoint':True})
                if 'inputs' in locals():
                    arrays={f'input{j}_{k}':v for j,item in enumerate(inputs) for k,v in item.items() if isinstance(v,np.ndarray)}
                    np.savez_compressed(folder/'failed_inputs.npz',**arrays)
                raise
            finally:reader.close()
            stage='sampling'
            for update in cfg['evaluation_updates']:
                path=checkpoints/f'pair_u{update:08d}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);body=dict(ck);h=body.pop('payload_digest')
                if p.digest(body)!=h or ck['binding']!=pair_binding or ck['updates']!=update:raise ValueError('evaluation checkpoint mismatch')
                for m in models:
                    m.load_state_dict(ck['models'][m.arm],strict=True);p.condition(m,target)
                    for sample_seed in cfg['sampling_seeds']:
                        check();dest=folder/f'samples_u{update:08d}'/m.arm/f'seed{sample_seed}';dest.parent.mkdir(parents=True,exist_ok=True)
                        if (dest/'summary.json').exists():
                            s=p.read(dest/'summary.json')
                            expected=dict(pair_binding,checkpoint_update=update,checkpoint_sha256=p.sha(path),arm=m.arm,sample_seed=sample_seed,model_digest=p.digest(m.state_dict()))
                            if s['status']!='completed' or s['binding']!=expected or s['graph_sha256']!=p.sha(dest/'graph.npz'):raise ValueError('existing sample evidence mismatch')
                            if not (dest/'comparison.json').exists():p.save(dest/'comparison.json',p.compare(p.read(dest/'metrics.json'),reference))
                            continue
                        metrics=chain(m,case,cfg,sample_seed,dest,dict(pair_binding,checkpoint_update=update,checkpoint_sha256=p.sha(path)),check)
                        p.save(dest/'comparison.json',p.compare(metrics,reference))
                        p.save(out/'progress.json',{'phase':'sampling','seed':seed,'checkpoint':update,'arm':m.arm,'sample_seed':sample_seed})
            completed.append(seed)
            for m in models:m.to('cpu')
            del models,opts
        # All raw graphs remain. WL-distinct hashes prove distinction; equal WL
        # hashes are NOT claimed isomorphic. Exact labelled duplicate and Jaccard
        # comparisons are descriptive, not a proof of population diversity.
        from evaluate import aggregate
        check();aggregate(out,cfg,check);status='completed'
    except (InterruptedError,TimeoutError) as exc:status='stopped_incomplete';error=repr(exc);raise
    except BaseException as exc:error=repr(exc);raise
    finally:p.save(out/'summary.json',{'status':status,'stage':stage,'error':error,'completed_train_seeds':completed,'wall_seconds_this_invocation':time.time()-start,
        'deadline_epoch':x.deadline,'budget_not_reset':True,'full_plan_updates':72000,'full_plan_chains':96,'plan_counts_not_claimed_completed':True,'no_guidance_or_repairs':True})
if __name__=='__main__':main()
