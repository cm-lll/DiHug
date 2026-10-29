"""Resume raw500 with original AdamW/EMA/RNG; fixed stage sampling."""
import argparse,json,os,pathlib,signal,time,traceback,resource,sys,copy
from runtime_contract import ROOT,read,save,sha,source_binding,verify_ticket,verify_readout,RESERVE,PROTOCOL,jobs,generation_checkpoints
sys.path.insert(0,str(ROOT/'package'))
import numpy as np
import torch
from model_io import configure
from resume_io import resume_parent
from worker import validation,atomic_checkpoint
from normal2 import normal2,pair_inputs
from probe import rng_get,input_hash
from hsgd.bw_preflight.common import schedule,state_edges,cpu_tree
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from evaluation import evaluate
from runtime_contract import DECAYS
from stage import isolated_readout
from integration import complete_normal2,checkpoint_payload,ema_validation

def paired_validation(model,ema,case,cfg,seed,rng):
    return {'raw':validation(model,case,cfg,seed,rng),'ema_0.99':ema_validation(validation,model,ema,case,cfg,seed,rng)}
def export_readouts(out,model,ema,updates,binding):
    receipt=dict(dataset=binding['dataset'],arm=binding['arm'],train_seed=binding['train_seed'],updates=updates,
        checkpoint_sha256=sha(out/f'checkpoint_u{updates}.pt'),files={},binding=binding)
    kind='ema_0.99';path=out/f'readout_u{updates}_ema_0.99.pt'
    atomic_checkpoint(path,dict(model=ema.readout_state_dict(model),readout=kind,decay=.99,updates=updates,
        train_seed=binding['train_seed'],binding=binding,checkpoint_sha256=receipt['checkpoint_sha256']))
    receipt['files'][kind]=dict(file=path.name,sha256=sha(path))
    save(out/f'readouts_u{updates}.json',receipt)

def train(job,out,approval,ledger,check,progress):
    model,opt,ema,case,cfg,rng,proof=resume_parent(job)
    binding=dict(source_sha256=source_binding()[1],protocol_sha256=sha(ROOT/'protocol.json'),formula_sha256=sha(ROOT/'formula.json'),
        dataset=job['dataset'],train_seed=job['train_seed'],arm=job['arm'],config=cfg,final_updates=job['updates'],
        resume_updates=500,parent=proof['parent'],parent_execution_commit='974e5dca8cbdd05ffddc53fd599ae815828d8b56',
        parent_result_commit='68a93bd2d6f39a71e6d525dac8647d9435df25ec',loss=PROTOCOL['loss'],
        lambda_count=model.count_aux_weight,approval_sha256=ledger['approval_sha256'])
    save(out/'binding.json',binding);save(out/'restore_receipt.json',proof);completed=500
    rows=[]
    def snapshot(name):
        begin=time.perf_counter()
        atomic_checkpoint(out/name,checkpoint_payload(model,opt,ema,completed,rng_get(rng,'cuda'),binding))
        with (out/'save_cost.jsonl').open('a') as f:f.write(json.dumps(dict(file=name,updates=completed,seconds=time.perf_counter()-begin))+'\n')
    try:
        snapshot('resumed_u500.pt')
        progress.update(status='restored',completed_updates=500,ema_updates=500,resumed_from=proof['parent']['sha256'])
        save(out/'progress.json',progress)
        for position in range(501,job['updates']+1):
            check();inputs=pair_inputs(case,cfg,position,rng);ih=input_hash(inputs)
            if position==501 and ih!=approval['next_input501_hashes'][job['id']]:raise ValueError('next input does not continue saved RNG')
            begin=time.perf_counter()
            rec=complete_normal2(normal2,model,opt,case,inputs,1000,'cuda',ema,completed);completed=position
            rec.update(update=completed,total_update_seconds=time.perf_counter()-begin,input_sha256=ih,t=[x['t'] for x in inputs])
            if rec.get('gradient_observation') is not None:raise ValueError('new gradient diagnostics not authorized')
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            progress.update(status='training',completed_updates=completed,ema_updates=ema.updates)
            save(out/'progress.json',progress)
            if completed in PROTOCOL['checkpoints']:
                check();snapshot(f'checkpoint_u{completed}.pt')
                rows.append(dict(updates=completed,**paired_validation(model,ema,case,cfg,job['train_seed'],rng)))
                save(out/'validation.json',rows)
            if completed in generation_checkpoints(job['dataset']):
                check();export_readouts(out,model,ema,completed,binding)
                # The existing isolation restores raw parameters/mode/RNG and
                # verifies optimizer and EMA remain exactly unchanged.
                with isolated_readout(model,opt,ema,rng,'cuda'):
                    for sj in jobs()['sampling']:
                        if sj['depends_on']!=job['id'] or sj['checkpoint']!=completed:continue
                        check();folder=pathlib.Path(ledger['output'])/sj['directory'];folder.mkdir(parents=True,exist_ok=False)
                        sp=dict(job=sj,status='running',completed_chains=0)
                        progress.update(status='sampling',sampling_checkpoint=completed,sampling_seed=sj['sample_seed'])
                        save(out/'progress.json',progress)
                        try:
                            sample(sj,folder,pathlib.Path(ledger['output']),approval,check,sp,loaded=(model,case,cfg))
                            sp['status']='complete';progress['completed_chains']+=1
                        except BaseException as exc:sp.update(status='incomplete',error=repr(exc),traceback=traceback.format_exc());raise
                        finally:save(folder/'summary.json',sp)
                save(out/f'sampling_isolation_u{completed}.json',dict(passed=True,raw_restored_exact=True,
                    optimizer_unchanged_exact=True,ema_unchanged_exact=True,training_rng_restored_exact=True,updates=completed,next_update=completed+1))
                progress.update(last_completed_generation_checkpoint=completed,training_state_restored=True)
                save(out/'progress.json',progress)
        check()
    except BaseException:
        if ema.at_update_boundary:
            try:snapshot('stopped_u%d.pt'%completed)
            except BaseException as exc:progress['stop_save_error']=repr(exc)
        else:progress['incomplete_update_not_saved']=True
        raise

def sample(job,out,run,approval,check,progress,loaded=None):
    path,receipt=verify_readout(run,job)
    payload=torch.load(path,map_location='cpu')
    if (payload['train_seed'],payload['readout'],payload['updates'],payload['checkpoint_sha256'])!=(job['train_seed'],job['readout'],job['checkpoint'],receipt['checkpoint_sha256']):raise ValueError('selected readout metadata mismatch')
    if payload.get('decay')!=DECAYS.get(job['readout']):raise ValueError('selected decay identity mismatch')
    if payload['binding']!=receipt['binding'] or payload['binding']['source_sha256']!=source_binding()[1]:raise ValueError('selected readout source')
    if loaded is None:raise RuntimeError('sampling only inside isolated trajectory')
    model,case,cfg=loaded
    model.load_state_dict(payload['model'],strict=True);model.eval()
    # Hash of actual loaded selected readout, never raw's hash substituted for EMA.
    from probe import equal
    if not equal(model.state_dict(),payload['model']):raise ValueError('readout load mismatch')
    del payload
    kernel=MarginalKernel(case['prior'],schedule(1000));families=np.repeat(np.arange(case['n_families']),case['space'].counts)
    sample_rng=np.random.default_rng(job['sample_seed']);state=(sample_rng.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
    initial_rng=copy.deepcopy(sample_rng.bit_generator.state);initial_state_sha=__import__('hashlib').sha256(state.tobytes()).hexdigest();next_t=1000;begin=time.perf_counter()
    def state_save(name):atomic_checkpoint(out/name,dict(state=state,next_t=next_t,rng=rng_get(sample_rng,'cuda'),job=job,model_sha256=sha(path)))
    try:
        for t in range(1000,0,-1):
            check();candidate,rec=reverse_step(model,case,state,kernel,t,1,job['sample_seed'],sample_rng,'cuda',check_time=check,trace=False)
            state=candidate;next_t=t-1
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if t%100==0 or t==1:state_save('complete_t%04d.pt'%next_t)
        check();edges,fam=state_edges(case,state)
        np.savez_compressed(out/'graph.npz',types=case['space'].types,edges=edges,families=fam,schema=np.array(case['space'].pairs))
        save(out/'metrics.json',evaluate(case,edges,cfg['datasets'][job['dataset']]['type_names'],check));check()
        save(out/'binding.json',dict(job=job,readout=job['readout'],model_sha256=sha(path),checkpoint_sha256=receipt['checkpoint_sha256'],
            graph_sha256=sha(out/'graph.npz'),metrics_sha256=sha(out/'metrics.json'),source_sha256=source_binding()[1],
            initial_state_sha256=initial_state_sha,initial_rng=initial_rng,seconds=time.perf_counter()-begin))
        progress['completed_chains']=1
    except BaseException:
        try:state_save('stopped_chain.pt')
        except BaseException as e:progress['chain_save_error']=repr(e)
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch-fd',type=int,required=True);p.add_argument('--approval',required=True);p.add_argument('--ledger',required=True);a=p.parse_args()
    ticket=json.loads(os.read(a.launch_fd,32768));os.close(a.launch_fd);job=ticket['job']
    if job not in jobs()['training']:raise ValueError('noncanonical train job')
    verify_ticket(ticket,job,a.approval,a.ledger,os.environ.get('CUDA_VISIBLE_DEVICES'),os.getppid())
    approval=read(a.approval);ledger=read(a.ledger);run=pathlib.Path(ledger['output']);out=run/job['directory'];out.mkdir(parents=True,exist_ok=False)
    stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
    def check():
        if stop[0] or (run/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('STOP/signal')
        if time.time()>=ledger['deadline_epoch']-RESERVE:raise TimeoutError('delivery reserve reached')
    progress=dict(status='initializing_restore',job=job,completed_updates=500,new_updates=0,completed_chains=0,started_epoch=time.time())
    try:
        check();env=configure()
        for k in ['python_version','torch_runtime_version','packages','device_name']:
            if env[k]!=approval['environment'][k]:raise ValueError('environment mismatch '+k)
        save(out/'environment.json',env);train(job,out,approval,ledger,check,progress);progress['status']='complete'
    except BaseException as exc:progress.update(status='incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        progress.update(finished_epoch=time.time(),new_updates=max(0,progress['completed_updates']-500),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():progress.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',progress)
    return 0 if progress['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())

