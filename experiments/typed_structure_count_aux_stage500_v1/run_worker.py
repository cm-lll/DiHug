"""Launcher-only stage500 paired training. This entry has no sampler."""
import argparse,json,os,pathlib,signal,time,traceback,resource,sys,functools
from runtime_contract import ROOT,read,save,sha,source_binding,verify_ticket,RESERVE,UPDATES,PROTOCOL,jobs
sys.path.insert(0,str(ROOT/'package'))
import torch
from model_io import configure,fresh
from worker import validation,atomic_checkpoint
from normal2 import normal2,pair_inputs
from probe import rng_get,input_hash
from ema import ParameterEMA
from integration import complete_normal2,checkpoint_payload,ema_validation

def paired_validation(model,ema,case,cfg,seed,rng):
    return {'raw':validation(model,case,cfg,seed,rng),'ema_0.99':ema_validation(validation,model,ema,case,cfg,seed,rng)}
def train(job,out,approval,ledger,check,progress):
    model,opt,case,cfg,rng,proof=fresh(job['dataset'],job['arm'],job['train_seed']);ema=ParameterEMA(model)
    binding=dict(source_sha256=source_binding()[1],protocol_sha256=sha(ROOT/'protocol.json'),
                 formula_sha256=sha(ROOT/'formula.json'),train_seed=job['train_seed'],
                 dataset=job['dataset'],arm=job['arm'],config=cfg,initial=proof,updates=UPDATES,
                 loss=PROTOCOL['loss'],lambda_count=model.count_aux_weight,approval_sha256=ledger['approval_sha256'])
    save(out/'binding.json',binding);completed=0;rows=[]
    def snapshot(name):
        begin=time.perf_counter()
        atomic_checkpoint(out/name,checkpoint_payload(model,opt,ema,completed,rng_get(rng,'cuda'),binding))
        with (out/'save_cost.jsonl').open('a') as f:f.write(json.dumps(dict(file=name,seconds=time.perf_counter()-begin))+'\n')
    try:
        snapshot('initial.pt')
        progress.update(status='initial_validation');save(out/'progress.json',progress)
        rows.append(dict(updates=0,**paired_validation(model,ema,case,cfg,job['train_seed'],rng)));save(out/'validation.json',rows)
        for position in range(1,UPDATES+1):
            check();inputs=pair_inputs(case,cfg,position,rng);begin=time.perf_counter()
            rec=complete_normal2(functools.partial(normal2,observe=position in PROTOCOL['gradient_observation_updates']),
                                model,opt,case,inputs,1000,'cuda',ema,completed);completed=position
            rec.update(update=completed,total_update_seconds=time.perf_counter()-begin,input_sha256=input_hash(inputs),t=[x['t'] for x in inputs])
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            if rec['gradient_observation'] is not None:
                save(out/f'gradient_u{completed}.json',dict(update=completed,input_sha256=rec['input_sha256'],t=rec['t'],actual_lambda=model.count_aux_weight,**rec['gradient_observation']))
            progress.update(status='training',completed_updates=completed,ema_updates=ema.updates);save(out/'progress.json',progress)
            if completed in PROTOCOL['checkpoints']:
                check();snapshot(f'checkpoint_u{completed}.pt')
                rows.append(dict(updates=completed,**paired_validation(model,ema,case,cfg,job['train_seed'],rng)));save(out/'validation.json',rows)
        check()
    except BaseException:
        if ema.at_update_boundary:
            try:snapshot('stopped_u%d.pt'%completed)
            except BaseException as exc:progress['stop_save_error']=repr(exc)
        else:progress['incomplete_update_not_saved']=True
        raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch-fd',type=int,required=True);p.add_argument('--approval',required=True);p.add_argument('--ledger',required=True);a=p.parse_args()
    ticket=json.loads(os.read(a.launch_fd,32768));os.close(a.launch_fd);job=ticket['job']
    if job not in jobs()['training'] or UPDATES!=500 or jobs()['sampling']:raise ValueError('stage500 canonical train job required')
    verify_ticket(ticket,job,a.approval,a.ledger,os.environ.get('CUDA_VISIBLE_DEVICES'),os.getppid())
    approval=read(a.approval);ledger=read(a.ledger);run=pathlib.Path(ledger['output']);out=run/job['directory'];out.mkdir(parents=True,exist_ok=False)
    stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
    def check():
        if stop[0] or (run/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('STOP/signal')
        if time.time()>=ledger['deadline_epoch']-RESERVE:raise TimeoutError('delivery reserve reached')
    progress=dict(status='initializing',job=job,completed_updates=0,completed_chains=0,started_epoch=time.time())
    try:
        check();env=configure()
        for k in ['python_version','torch_runtime_version','packages','device_name']:
            if env[k]!=approval['environment'][k]:raise ValueError('environment mismatch '+k)
        save(out/'environment.json',env);train(job,out,approval,ledger,check,progress);progress['status']='complete'
    except BaseException as exc:progress.update(status='incomplete',error=repr(exc),traceback=traceback.format_exc())
    finally:
        progress.update(finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():progress.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',progress)
    return 0 if progress['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
