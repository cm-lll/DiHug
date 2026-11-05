"""Four registered candidates only: initial -> 3000, then four paired T1000 chains."""
from .formal_runtime import *
import argparse,signal,traceback,gc
from .formal_sampling import sample_one

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=TASKS,required=True);a=ap.parse_args()
    setup();authenticate_sources()
    if not read(OUT/'budget.json')['fits'] or not read(OUT/'engineering/receipt.json')['passed']:raise PermissionError('engineering/budget gate')
    approval=read(OUT/'approval.json')
    if not approval['authorized'] or os.getppid()!=approval['launcher_pid']:raise PermissionError('launch binding')
    folder=OUT/'run'/a.task;folder.mkdir(parents=True,exist_ok=False)
    status=dict(task=a.task,status='loading',updates=0,chains=0,started=now(),pid=os.getpid(),device=os.environ['CUDA_VISIBLE_DEVICES'])
    boundary=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    def check_local():
        if stopping:raise InterruptedError('own worker stop requested')
        check()
    def event(**kw):status.update(kw,at=now());save(folder/'progress.json',status)
    try:
        row,case,cfg,m,o,e,r,boundary,binding=load(a.task)
        if str(row['gpu'])!=os.environ['CUDA_VISIBLE_DEVICES']:raise ValueError('device identity')
        save(folder/'binding.json',binding);save(folder/'environment.json',g.common.environment('cuda'))
        atomic_checkpoint(folder/'initial.pt',boundary.checkpoint(binding))
        vals=[]
        def val(u):
            with boundary.validation():
                value=dict(updates=u,raw=validation(m,case,cfg,row['train_seed'],r),ema_0_99=ema_validation(validation,m,e,case,cfg,row['train_seed'],r,boundary=boundary))
            vals.append(value);save(folder/'validation.json',vals)
        val(0);contracts=read(OUT/'prepared'/f"input_contract_{row['dataset']}_{row['train_seed']}.json")
        baseline=[json.loads(x) for x in (BASE/'run'/a.task/'training.jsonl').read_text().splitlines()]
        for pos in range(1,3001):
            check_local();tick=time.perf_counter()
            with boundary.step(pos):
                inp=g.one_input(case,cfg,pos,r);ih=input_hash([inp]);expected=contracts[pos-1]
                if ih!=expected['sha256'] or inp['t']!=expected['t'] or len(inp['ids'])!=expected['slots']:raise ValueError('main input mismatch '+str(pos))
                pre_rng=g.digest(g.rng_capture(r,'cuda'))
                rec=g.update(m,o,case,inp,1000,'cuda',e,boundary.completed,transaction=boundary)
                post_rng=g.digest(g.rng_capture(r,'cuda'))
                if pre_rng!=post_rng:raise ValueError('candidate compute unexpectedly consumed training RNG')
            rec.update(update=pos,t=inp['t'],input_sha256=ih,paired_input_exact=True,full_rng_before=pre_rng,full_rng_after=post_rng,at=now(),total_update_seconds=time.perf_counter()-tick)
            with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            if pos%5==0 or pos==1:event(status='training',updates=pos,loss=rec['loss'],seconds=rec['total_update_seconds'])
            if pos in (1000,2000,3000):
                atomic_checkpoint(folder/('final.pt' if pos==3000 else f'checkpoint_u{pos}.pt'),boundary.checkpoint(binding));val(pos)
        states=final_readouts(m,e,3000);cp=sha(folder/'final.pt')
        for kind,sd in [('raw',states['raw']),('ema_0.99',states['ema'])]:
            atomic_checkpoint(folder/f'readout_{kind}.pt',dict(model=sd,readout=kind,decay=.99 if kind!='raw' else None,updates=3000,train_seed=row['train_seed'],binding=binding,checkpoint_sha256=cp))
        boundary=None;del m,o,e,r,states;gc.collect();torch.cuda.empty_cache()
        registration=dict(id=a.task,dataset=row['dataset'],train_seed=row['train_seed'],configuration='STRUCTURE_late',readout_file=str(folder/'readout_ema_0.99.pt'),readout_sha256=sha(folder/'readout_ema_0.99.pt'),parent_approval_sha256=sha(OUT/'approval.json'))
        save(folder/'sampling_registration.json',registration)
        for ss in SAMPLES:
            check_local();event(status='sampling',sample_seed=ss)
            sample_one(registration,ss,folder/'ema_0.99'/f'sample_{ss}',case,cfg,None,OUT,check_local,event)
            status['chains']+=1
        event(status='complete')
    except BaseException as exc:
        event(status='failed_or_incomplete',error=repr(exc),traceback=traceback.format_exc())
        if boundary is not None:
            status['updates']=boundary.completed;status['recovery']=boundary.last_failure
            if boundary.resumable and boundary.phase=='idle':atomic_checkpoint(folder/f'stopped_u{boundary.completed}.pt',boundary.checkpoint(binding))
            else:status['exact_resume_allowed']=False
        raise
    finally:
        save(folder/'summary.json',dict(status,finished=now(),peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

if __name__=='__main__':main()
