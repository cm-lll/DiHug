"""Adapted frozen O worker: same normal2/validation/reverse_step/evaluate calls.

Train once, export two independent final readouts; each chain authenticates and
loads exactly its selected file. Only launcher-issued pipes admit formal jobs.
"""
import argparse,json,os,pathlib,signal,time,traceback,resource,copy,sys
from runtime_contract import ROOT,read,save,sha,source_binding,verify_ticket,verify_readout,cell,RESERVE
sys.path.insert(0,str(ROOT/'package'))
import numpy as np
import torch
from model_io import configure,fresh
from worker import validation,atomic_checkpoint
from normal2 import normal2,pair_inputs
from probe import rng_get,input_hash
from hsgd.bw_preflight.common import schedule,state_edges,cpu_tree
from hsgd.kernel import MarginalKernel
from hsgd.runtime import reverse_step
from evaluation import evaluate
from ema import ParameterEMA
from integration import complete_normal2,checkpoint_payload,ema_validation,final_readouts

def paired_validation(model,ema,case,cfg,seed,rng):
    return dict(raw=validation(model,case,cfg,seed,rng),ema=ema_validation(validation,model,ema,case,cfg,seed,rng))

def export_readouts(out,model,ema,updates,binding):
    states=final_readouts(model,ema,updates);receipt=dict(train_seed=binding['train_seed'],updates=updates,
        checkpoint_sha256=sha(out/'final.pt'),files={},binding=binding)
    for kind,state in states.items():
        path=out/('readout_'+kind+'.pt')
        atomic_checkpoint(path,dict(model=state,readout=kind,updates=updates,train_seed=binding['train_seed'],
                                   binding=binding,checkpoint_sha256=receipt['checkpoint_sha256']))
        receipt['files'][kind]=dict(file=path.name,sha256=sha(path))
    save(out/'readouts.json',receipt)

def train(job,out,approval,ledger,check,progress):
    model,opt,case,cfg,rng,proof=fresh(job['train_seed'],approval.get('initial_paths'));ema=ParameterEMA(model)
    binding=dict(source_sha256=source_binding()[1],protocol_sha256=sha(ROOT/'protocol.json'),train_seed=job['train_seed'],
                 dataset='PubMed',arm='O',config=cfg,initial=proof,updates=1000,approval_sha256=ledger['approval_sha256'])
    save(out/'binding.json',binding)
    completed=0
    def snapshot(name):atomic_checkpoint(out/name,checkpoint_payload(model,opt,ema,completed,rng_get(rng,'cuda'),binding))
    # theta0, fresh empty optimizer, all RNG saved BEFORE any forward.
    snapshot('initial.pt');rows=[dict(updates=0,**paired_validation(model,ema,case,cfg,job['train_seed'],rng))]
    save(out/'validation.json',rows)
    try:
        for position in range(1,1001):
            check();inputs=pair_inputs(case,cfg,position,rng);begin=time.perf_counter()
            rec=complete_normal2(normal2,model,opt,case,inputs,1000,'cuda',ema,completed);completed=position
            rec.update(update=completed,total_update_seconds=time.perf_counter()-begin,input_sha256=input_hash(inputs),t=[x['t'] for x in inputs])
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            progress.update(completed_updates=completed,ema_updates=ema.updates);save(out/'progress.json',progress)
            if completed in [500,1000]:
                check();snapshot('midpoint.pt' if completed==500 else 'final.pt')
                rows.append(dict(updates=completed,**paired_validation(model,ema,case,cfg,job['train_seed'],rng)));save(out/'validation.json',rows)
        check();export_readouts(out,model,ema,completed,binding)
    except BaseException:
        if ema.at_update_boundary:
            try:snapshot('stopped_u%d.pt'%completed)
            except BaseException as e:progress['stop_save_error']=repr(e)
        else:progress['incomplete_update_not_saved']=True
        raise

def sample(job,out,run,approval,check,progress):
    path,receipt=verify_readout(run,job)
    payload=torch.load(path,map_location='cpu')
    if (payload['train_seed'],payload['readout'],payload['updates'],payload['checkpoint_sha256'])!=(job['train_seed'],job['readout'],1000,receipt['checkpoint_sha256']):raise ValueError('selected readout metadata mismatch')
    if payload['binding']!=receipt['binding'] or payload['binding']['source_sha256']!=source_binding()[1]:raise ValueError('selected readout source')
    model,opt,case,cfg,unused,proof=fresh(job['train_seed'],approval.get('initial_paths'));del opt,unused
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
        save(out/'metrics.json',evaluate(case,edges,cfg['datasets']['PubMed']['type_names'],check));check()
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
    from runtime_contract import jobs
    if job not in jobs()['training']+jobs()['sampling']:raise ValueError('noncanonical job')
    verify_ticket(ticket,job,a.approval,a.ledger,os.environ.get('CUDA_VISIBLE_DEVICES'),os.getppid())
    approval=read(a.approval);ledger=read(a.ledger);run=pathlib.Path(ledger['output']);out=run/job['directory'];out.mkdir(parents=True,exist_ok=False)
    stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,lambda s,f:stop.__setitem__(0,True))
    def check():
        if stop[0] or (run/'STOP').exists() or (out/'STOP').exists():raise InterruptedError('STOP/signal')
        if time.time()>=ledger['deadline_epoch']-RESERVE:raise TimeoutError('delivery reserve reached')
    progress=dict(status='running',job=job,completed_updates=0,completed_chains=0,started_epoch=time.time())
    try:
        check();env=configure()
        for k in ['python_version','torch_runtime_version','packages','device_name']:
            if env[k]!=approval['environment'][k]:raise ValueError('environment mismatch '+k)
        save(out/'environment.json',env)
        if job['mode']=='train':train(job,out,approval,ledger,check,progress)
        else:sample(job,out,run,approval,check,progress)
        progress['status']='complete'
    except BaseException as e:progress.update(status='incomplete',error=repr(e),traceback=traceback.format_exc())
    finally:
        progress.update(finished_epoch=time.time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if torch.cuda.is_initialized():progress.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(out/'summary.json',progress)
    return 0 if progress['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())
