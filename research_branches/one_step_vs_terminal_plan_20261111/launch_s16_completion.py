"""Resume only the two stopped S16 arms; own-process budget supervisor."""
import os,time,json,sys,hashlib,signal,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/three_arm_s16_completion_20261111T232800'
OLD=ROOT/'results/three_arm_learning_pilot_20261111T211641'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
# Conservative start includes preparation in this newly authorized turn.
START=datetime.datetime.fromisoformat('2026-11-11T23:28:00+08:00').timestamp()
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
    p=json.loads((OLD/'protocol.json').read_text());old_release=json.loads((OLD/'release.json').read_text())
    assert json.loads((OLD/'failure_audit/solver_regression.json').read_text())['passed']
    prior={r['dataset']:r['seconds'] for r in old_release['processes'] if r['arm']=='S16'}
    remaining={ds:p['training_seconds'][ds]-prior[ds] for ds in prior}
    assert all(v>0 for v in remaining.values())
    next_inputs={r['dataset']:{k:r[k] for k in ('t','input_hash')} for r in json.loads((OLD/'failure_audit/reconstruction.json').read_text())}
    bindings=p['bindings'].copy()
    for name in ('resume_s16_worker.py','launch_s16_completion.py','evaluate_s16_completion.py','finalize_s16_completion.py','heat_solver_candidate.py'):
        f=Path(__file__).with_name(name);bindings[str(f)]=sha(f)
    for ds in prior:
        for name in ('stopped_complete.pt','binding.json','updates.jsonl'):
            f=OLD/(ds+'_S16')/name;bindings[str(f)]=sha(f)
    for path,h in bindings.items():assert sha(path)==h,path
    # Observed chain times 480/650s; add 120s per-task headroom, includes restore/validation.
    nominal=max(remaining['IMDB']+960,remaining['PubMed']+1300)+120
    if time.time()+nominal>=START+2100:
        save('blocked.json',dict(reason='complete remaining training plus four full chains no longer fits science window',remaining_science_seconds=START+2100-time.time(),estimated_need_seconds=nominal));return
    p.update(start_epoch=START,science_stop_epoch=START+2100,delivery_epoch=START+2400,GPU_hours=1.2,max_GPUs=2,
        bindings=bindings,resume_parent=str(OLD),resume_updates={'IMDB':749,'PubMed':1180},resume_next_inputs=next_inputs,
        prior_allocation_seconds=prior,training_seconds=remaining,prior_GPU_hours=old_release['GPU_hours'],
        prior_protocol_sha256=sha(OLD/'protocol.json'),new_evaluation_chains=4,reused_evaluation_graphs=12,
        stopping='35min science /40min delivery /1.2 allocated GPUh first; no successor',
        score_repair='same heat operator/scaling, evd instead of evr; same original tolerances plus residual checks',
        resume_semantics='full saved model/AdamW/EMA/RNG/counters; no retraining or reset; old used card time deducted',
        forecast_seconds=nominal,forecast_GPU_hours=(sum(remaining.values())+960+1300+240)/3600)
    save('protocol.json',p)
    for ds in prior:
        for arm in ('baseline','S1','T_eager'):(OUT/(ds+'_'+arm)).symlink_to(OLD/(ds+'_'+arm),target_is_directory=True)
    procs=[]
    for gpu,ds in zip(free[:2],('IMDB','PubMed')):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[k]='1'
        at=time.time();env['TASK_ALLOCATION_EPOCH']=str(at)
        log=(OUT/(ds+'_S16.log')).open('w')
        proc=subprocess.Popen([PY,str(Path(__file__).with_name('resume_s16_worker.py')),str(OUT),ds,'S16'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(dict(proc=proc,dataset=ds,gpu=gpu,start=at,end=None,log=log))
    save('launch.json',[dict(dataset=x['dataset'],arm='S16',gpu=x['gpu'],pid=x['proc'].pid,start_epoch=x['start']) for x in procs])
    final_log=(OUT/'finalizer.log').open('w')
    final=subprocess.Popen([PY,str(Path(__file__).with_name('finalize_s16_completion.py')),str(OUT)],stdout=final_log,stderr=subprocess.STDOUT,start_new_session=True)
    save('finalizer_binding.json',dict(pid=final.pid,CPU_only=True));final_log.close()
    stopped=False
    while True:
        now=time.time()
        for x in procs:
            if x['end'] is None and x['proc'].poll() is not None:x['end']=now
        hours=sum((x['end'] or now)-x['start'] for x in procs)/3600
        save('watchdog.json',dict(at_epoch=now,GPU_hours=hours,science_remaining_seconds=START+2100-now,processes=[dict(dataset=x['dataset'],pid=x['proc'].pid,returncode=x['proc'].poll()) for x in procs]))
        if all(x['end'] is not None for x in procs):break
        if not stopped and (now>=START+2090 or hours>=1.19):
            stopped=True
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
            time.sleep(3)
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGKILL)
        time.sleep(1)
    for x in procs:x['log'].close()
    save('release.json',dict(wall_seconds=time.time()-START,GPU_hours=hours,stopped_at_limit=stopped,processes=[dict(dataset=x['dataset'],arm='S16',gpu=x['gpu'],pid=x['proc'].pid,returncode=x['proc'].returncode,seconds=x['end']-x['start']) for x in procs]))
    (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
if __name__=='__main__':main()
