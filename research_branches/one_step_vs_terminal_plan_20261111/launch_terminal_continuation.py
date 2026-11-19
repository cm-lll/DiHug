"""Two T_eager continuations; allocated-card budget and hard wall supervisor."""
import os,time,json,sys,hashlib,signal,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/terminal_continuation_20261112T003000'
OLD=ROOT/'results/three_arm_learning_pilot_20261111T211641'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
# Conservative start includes preparation in this newly authorized turn.
START=datetime.datetime.fromisoformat('2026-11-12T00:30:00+08:00').timestamp()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def main():
    OUT.mkdir(exist_ok=True)
    lock=os.open(OUT/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(lock)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    (OUT/'gpu_launch_check.txt').write_text(rows)
    free=[r.split(',')[0].strip() for r in rows.splitlines() if int(r.split(',')[2])==0 and int(r.split(',')[3])==0]
    assert len(free)>=2, 'two empty GPUs required'
    plan_path=ROOT/'results/terminal_continuation_plan_20261112/plan.json'
    plan=json.loads(plan_path.read_text());p=json.loads((OLD/'protocol.json').read_text())
    assert json.loads((OLD/'failure_audit/solver_regression.json').read_text())['passed']
    assert plan['final_updates']==8 and plan['new_training_chains']==24 and plan['new_evaluation_chains']==8
    bindings=p['bindings'].copy();bindings.update(plan['bindings']);bindings[str(plan_path)]=sha(plan_path)
    for name in ('terminal_continuation_worker.py','launch_terminal_continuation.py','evaluate_terminal_continuation.py','finalize_terminal_continuation.py','heat_solver_candidate.py'):
        f=Path(__file__).with_name(name);bindings[str(f)]=sha(f)
    for path,h in bindings.items():assert sha(path)==h,path
    nominal=plan['nominal_parallel_wall_minutes']*60
    if time.time()+nominal+300>=START+12000:
        save('blocked.json',dict(reason='complete panel plus five-minute headroom does not fit science window',remaining_seconds=START+12000-time.time(),need_seconds=nominal+300));return
    p.update(start_epoch=START,science_stop_epoch=START+12000,delivery_epoch=START+13200,GPU_hours=6.5,max_GPUs=2,
        bindings=bindings,resume_parent=str(OLD),resume_updates={'IMDB':2,'PubMed':2},
        final_updates=8,evaluate_at_updates=[4,8],terminal_plan=plan['terminal_plan'],
        arms=['T_eager'],terminal_rounds=8,new_terminal_rounds_per_dataset=6,
        weights='restore complete original T_eager update2; preserve AdamW and EMA',
        training_chain_count=24,new_evaluation_chains=8,reused_evaluation_graphs=8,
        stopping='200min science /220min delivery /6.5 allocated GPUh first; no successor',
        score_repair='same heat/scaling, evd with unchanged original tolerances plus residual checks',
        resume_semantics='full final update2 model/AdamW/EMA/RNG/counts, no optimizer reset',
        forecast_seconds=nominal,forecast_GPU_hours=plan['nominal_GPU_hours'],approved_plan_sha256=sha(plan_path))
    save('protocol.json',p)
    for ds in ('IMDB','PubMed'):
        (OUT/(ds+'_baseline')).symlink_to(OLD/(ds+'_baseline'),target_is_directory=True)
        (OUT/(ds+'_T2')).symlink_to(OLD/(ds+'_T_eager'),target_is_directory=True)
        for stage in (4,8):(OUT/(ds+f'_T{stage}')).symlink_to(OUT/(ds+'_T_eager')/f'stage_{stage}',target_is_directory=True)
    procs=[]
    for gpu,ds in zip(free[:2],('IMDB','PubMed')):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
        at=time.time();env['TASK_ALLOCATION_EPOCH']=str(at)
        log=(OUT/(ds+'_T_eager.log')).open('w')
        proc=subprocess.Popen([PY,str(Path(__file__).with_name('terminal_continuation_worker.py')),str(OUT),ds,'T_eager'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(dict(proc=proc,dataset=ds,gpu=gpu,start=at,end=None,log=log))
    save('launch.json',[dict(dataset=x['dataset'],arm='T_eager',gpu=x['gpu'],pid=x['proc'].pid,start_epoch=x['start']) for x in procs])
    final_log=(OUT/'finalizer.log').open('w')
    final=subprocess.Popen([PY,str(Path(__file__).with_name('finalize_terminal_continuation.py')),str(OUT)],stdout=final_log,stderr=subprocess.STDOUT,start_new_session=True)
    save('finalizer_binding.json',dict(pid=final.pid,CPU_only=True));final_log.close()
    stopped=False
    while True:
        now=time.time()
        for x in procs:
            if x['end'] is None and x['proc'].poll() is not None:x['end']=now
        hours=sum((x['end'] or now)-x['start'] for x in procs)/3600
        save('watchdog.json',dict(at_epoch=now,GPU_hours=hours,science_remaining_seconds=START+12000-now,processes=[dict(dataset=x['dataset'],pid=x['proc'].pid,returncode=x['proc'].poll()) for x in procs]))
        if all(x['end'] is not None for x in procs):break
        if not stopped and (now>=START+11990 or hours>=6.49):
            stopped=True
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
            time.sleep(3)
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGKILL)
        time.sleep(1)
    for x in procs:x['log'].close()
    save('release.json',dict(wall_seconds=time.time()-START,GPU_hours=hours,stopped_at_limit=stopped,processes=[dict(dataset=x['dataset'],arm='T_eager',gpu=x['gpu'],pid=x['proc'].pid,returncode=x['proc'].returncode,seconds=x['end']-x['start']) for x in procs]))
    (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
if __name__=='__main__':main()
