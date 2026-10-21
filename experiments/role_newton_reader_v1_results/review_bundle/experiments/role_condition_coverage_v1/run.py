"""One inherited three-hour ledger, six isolated CPU jobs, gated formal learning."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def save(p,x):
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true',help='use existing unique3h ledger; gate then full fixed panel')
    args=parser.parse_args()
    if not args.execute:parser.error('--execute is required; no implicit learning')
    from common import (RESULT,EXP,read,settings,check,check_binding,sha,source_manifest)
    cfg=settings();activation=read(RESULT/'activation.json');jobs=[];stage='probe';status='failed';reason=None
    if (RESULT/'formal_launch.json').exists():raise FileExistsError('single invocation; no reset/retry launch')
    save(RESULT/'formal_launch.json',dict(time_epoch=time.time(),activation=activation,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()))
    def interrupt(sig,frame):(RESULT/'STOP').touch()
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    def launch(mode,arm,seed):
        folder=RESULT/('cost' if mode=='probe' else 'jobs')/f'{arm}_{seed}'
        folder.parent.mkdir(exist_ok=True)
        logfile=RESULT/f'{mode}_{arm}_{seed}.log';stream=logfile.open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),mode,'--arm',arm,'--seed',str(seed),'--folder',str(folder)],
             stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        job=dict(process=child,folder=folder,log=stream,mode=mode,arm=arm,seed=seed);jobs.append(job);return job
    def wait(active):
        last=0
        while any(j['process'].poll() is None for j in active):
            check()
            for j in active:
                code=j['process'].poll()
                if code not in (0,None):raise RuntimeError(f"{stage} {j['arm']} {j['seed']} exit{code}; see saved log")
            if time.monotonic()-last>20:
                rows=[]
                for j in active:
                    f=j['folder']/'progress.json';rows.append(dict(arm=j['arm'],seed=j['seed'],exit_code=j['process'].poll(),progress=read(f) if f.exists() else None))
                save(RESULT/'progress.json',dict(stage=stage,remaining_seconds=activation['deadline_epoch']-time.time(),jobs=rows))
                print(json.dumps(dict(stage=stage,jobs=rows)),flush=True);last=time.monotonic()
            time.sleep(.5)
        for j in active:
            assert j['process'].wait()==0
            receipt=read(j['folder']/(j['mode']+'_receipt.json'));assert receipt['status']=='completed'
            j['log'].close()
    try:
        check();check_binding();assert len(os.sched_getaffinity(0))>=6
        active=[launch('probe',arm,seed) for arm in cfg['arms'] for seed in cfg['train_seeds']]
        wait(active);check()
        probes=[read(j['folder']/'probe.json') for j in active]
        worst_update=max(p['training_worst_P90'] for p in probes);worst_sweep=max(p['sweep_worst_seconds'] for p in probes)
        train=6000*worst_update;sample=88*64*worst_sweep
        validation=len(cfg['validation_updates'])*max(p['validation_seconds'] for p in probes)
        saves=5*max(p['save_reload_seconds'] for p in probes)
        predicted=cfg['cost_safety_factor']*(train+sample+validation+saves+120)+cfg['audit_delivery_reserve_seconds']
        remaining=activation['deadline_epoch']-time.time();passed=predicted<=.8*remaining
        save(RESULT/'cost_gate.json',dict(passed=passed,remaining_seconds=remaining,allowed80percent_seconds=.8*remaining,
             forecast_wall_seconds=predicted,training_wall_seconds=train,sampling528_wall_seconds=sample,
             validation_wall_seconds=validation,save_reload_wall_seconds=saves,
             safety_factor=cfg['cost_safety_factor'],audit_delivery_reserve_seconds=cfg['audit_delivery_reserve_seconds'],
             CPU_parallel_jobs=6,full_model_conditions_considered=True,probes=probes))
        if not passed:
            status='cost_blocked';reason='all fixed work exceeds80percent of remaining original budget; no scientific updates';return 3
        save(RESULT/'learning_authorization.json',dict(cost_gate_sha256=sha(RESULT/'cost_gate.json'),activation_sha256=sha(RESULT/'activation.json'),time_epoch=time.time(),updates=36000,chains=528))
        stage='training';active=[launch('train',arm,seed) for arm in cfg['arms'] for seed in cfg['train_seeds']]
        wait(active);check();check_binding()
        weights=[read(j['folder']/'final_weight.json') for j in active]
        assert len(weights)==6 and all(w['completed_updates']==6000 and sha(RESULT/w['path'])==w['sha256'] for w in weights)
        save(RESULT/'sealed_final_weights.json',dict(all_six_complete=True,weights=weights,time_epoch=time.time(),
             unseen_test_results_viewed=False,selection='all final6000, no checkpoint selection'))
        stage='sampling';active=[launch('sample',arm,seed) for arm in cfg['arms'] for seed in cfg['train_seeds']]
        wait(active);check();check_binding()
        stage='audit'
        from coverage_evaluation import audit
        audit();check();check_binding();status='completed'
    except BaseException as exc:
        reason=repr(exc);traceback.print_exc();(RESULT/'STOP').touch()
        status='stopped' if isinstance(exc,(InterruptedError,TimeoutError)) else 'failed'
    finally:
        for j in jobs:
            if j['process'].poll() is None:
                try:os.killpg(j['process'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        cleanup=time.monotonic()
        for j in jobs:
            try:j['process'].wait(timeout=max(.1,15-(time.monotonic()-cleanup)))
            except subprocess.TimeoutExpired:
                try:os.killpg(j['process'].pid,signal.SIGKILL)
                except ProcessLookupError:pass
                j['process'].wait()
            j['log'].close()
        save(RESULT/'controller.json',dict(status=status,stage=stage,reason=reason,elapsed_from_original_activation=time.time()-activation['activation_epoch'],
             budget_seconds=10800,activation_preserved=True,cleanup_seconds=time.monotonic()-cleanup,
             scientific_updates_recorded=sum(sum(1 for _ in p.open()) for p in RESULT.glob('jobs/*/updates.jsonl')),
             scientific_graphs_recorded=len(list(RESULT.glob('jobs/*/generated/*_result.json')))))
    return 0 if status=='completed' else 1
if __name__=='__main__':sys.exit(main())
