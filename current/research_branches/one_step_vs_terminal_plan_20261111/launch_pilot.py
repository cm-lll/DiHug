"""Own-only supervisor: 90min total, 80min science, 8.5 allocated GPU-hours.

The window begins at preparation, not worker launch. No restart resets it.
"""
import os,time,json,sys,hashlib,signal,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/three_arm_learning_pilot_20261111T211641'
PY=sys.executable
START=datetime.datetime.fromisoformat('2026-11-11T21:16:41+08:00').timestamp()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'launch.json').exists(), 'Never duplicate launched workers'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    (OUT/'gpu_launch_check.txt').write_text(rows)
    free=[r.split(',')[0].strip() for r in rows.splitlines() if int(r.split(',')[2])==0 and int(r.split(',')[3])==0]
    if len(free)<7:save('blocked_recheck.json',dict(reason='six simultaneous training tasks plus one sequential baseline device required',actual_free=free));return
    old=json.loads((ROOT/'results/three_arm_update_preflight_20261111/protocol.json').read_text())
    bindings=old['bindings'].copy()
    parents=json.loads((ROOT/'results/terminal_structure_real_20261109T233600/protocol.json').read_text())['models']
    proposal=json.loads((ROOT/'results/three_arm_update_preflight_20261111/pilot_budget_proposal.json').read_text())
    for p in [Path(__file__),Path(__file__).with_name('pilot_worker.py'),
              ROOT/'results/three_arm_update_preflight_20261111/pilot_budget_proposal.json']:
        bindings[str(p)]=sha(p)
    for ds in parents:
        for name in ('graph.npz','identity.json','steps.jsonl','full_metrics.json'):
            p=ROOT/f'results/three_arm_update_preflight_20261111/{ds}/baseline_sample_2026111341/{name}'
            bindings[str(p)]=sha(p)
    for p,h in bindings.items():assert sha(p)==h,p
    # This launcher only constructs CPU seed schedules, not models.
    import numpy as np
    terminal_plan={}
    for ds,row in parents.items():
        terminal_plan[ds]={}
        for round_id in (1,2):
            terminal_plan[ds][str(round_id)]=[]
            for k in (0,1):
                rng=np.random.default_rng(np.random.SeedSequence([2026111505,row['train_seed'],round_id,k]))
                times=sorted(map(int,rng.choice(np.arange(1,1001),size=8,replace=False)))
                terminal_plan[ds][str(round_id)].append(dict(seed=2026111501+(round_id-1)*2+k,times=times))
    remaining=(START+4800-time.time())/60
    if remaining<proposal['six_worker_nominal_wall_minutes']+5:
        save('blocked.json',dict(reason='remaining science window cannot cover full nominal panel plus minimum five-minute headroom',remaining_minutes=remaining));return
    protocol=dict(start_epoch=START,science_stop_epoch=START+4800,delivery_epoch=START+5400,GPU_hours=8.5,max_GPUs=8,
        bindings=bindings,parents=parents,training_seconds=proposal['matched_training_GPU_seconds_per_seed'],terminal_plan=terminal_plan,
        arms=['S1','S16','T_eager'],sample_seeds=[2026111341,2026111342],baseline_reuse_seed=2026111341,
        terminal_rounds=2,terminal_K=2,selected_times_per_trajectory=8,training_chain_count=8,new_evaluation_chains=14,
        optimizer_lr=1e-5,EMA=.99,gradient_clip=1,loss_scale=.01,model_mode='eval, same tested inactive-dropout production path',
        weights='same pretrained A late-CN EMA3000; new AdamW and EMA; remaining optimizer settings from parent',
        objective='typed joint heat-degree kernel MMD + per-relation count risk, unchanged frozen reference scales',
        S1='one hard clean graph, raw cost score-function, no sample baseline',
        S16='all16 draws: a_k=(16*C_k-sum(C))/15; mean(a_k*sum_edge_log_probability), no winner selection',
        T_eager='two fresh complete trajectories per frozen-policy round; terminal LOO advantages; mean over16 clean-head selected-time VJPs; biased surrogate',
        primary_readout='fixed final raw; EMA saved and separately validated',
        input_roots=dict(main=2026111401,hard=2026111404,validation=2026111511),
        independent_endpoint_validation_times=[1,75,250,550,800,1000],
        engineering_gates_reused='previous12 temporary updates/recovery +2fullchains; no short quality improvement gate',
        frozen_quality_interpretation='report all raw/LCC metrics and relative changes; no best checkpoint/seed selection; one training seed per dataset and two outputs are screening only',
        stopping='any science wall/8.5GPUh first; complete boundaries saved; never shorten chains/relabel incomplete arms',
        nominal_GPU_hours=proposal['estimated_training_GPU_hours']+proposal['estimated_final_generation_GPU_hours'],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save('protocol.json',protocol)
    tasks=[(ds,arm) for ds in ('IMDB','PubMed') for arm in ('S1','S16','T_eager')]+[(ds,'baseline') for ds in ('IMDB','PubMed')]
    procs=[];pending=tasks.copy()
    def launch(gpu,task):
        ds,arm=task
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]='1'
        log=(OUT/f'{ds}_{arm}.log').open('w');allocation=time.time()
        p=subprocess.Popen([PY,str(Path(__file__).with_name('pilot_worker.py')),str(OUT),ds,arm],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(dict(proc=p,gpu=gpu,dataset=ds,arm=arm,start=allocation,end=None,log=log))
    for gpu in free:
        if pending:launch(gpu,pending.pop(0))
    save('launch.json',[dict(pid=x['proc'].pid,gpu=x['gpu'],dataset=x['dataset'],arm=x['arm'],start_epoch=x['start']) for x in procs])
    stopped=False
    while True:
        now=time.time()
        for x in procs:
            if x['end'] is None and x['proc'].poll() is not None:x['end']=now
        if pending and not stopped and now<START+4790:
            busy={x['gpu'] for x in procs if x['end'] is None}
            for gpu in free:
                if gpu not in busy and pending:launch(gpu,pending.pop(0))
            save('launch.json',[dict(pid=x['proc'].pid,gpu=x['gpu'],dataset=x['dataset'],arm=x['arm'],start_epoch=x['start']) for x in procs])
        hours=sum((x['end'] or now)-x['start'] for x in procs)/3600
        save('watchdog.json',dict(at_epoch=now,GPU_hours=hours,science_remaining_seconds=START+4800-now,
            processes=[dict(pid=x['proc'].pid,dataset=x['dataset'],arm=x['arm'],returncode=x['proc'].poll()) for x in procs]))
        if all(x['end'] is not None for x in procs):break
        if not stopped and (now>=START+4790 or hours>=8.475):
            stopped=True
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
            time.sleep(3)
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGKILL)
        time.sleep(1)
    for x in procs:x['log'].close()
    save('release.json',dict(wall_seconds=time.time()-START,GPU_hours=hours,stopped_at_limit=stopped,
        processes=[dict(dataset=x['dataset'],arm=x['arm'],gpu=x['gpu'],pid=x['proc'].pid,returncode=x['proc'].returncode,seconds=x['end']-x['start']) for x in procs]))
    (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
    print((OUT/'release.json').read_text(),flush=True)
if __name__=='__main__':main()
