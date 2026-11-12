"""Single owned worker, 15-minute/.25 GPU-hour hard envelope."""
import os,sys,json,time,subprocess,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/one_step_terminal_gpu_preflight_20261111'
PY=sys.executable
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2))
def main():
    OUT.mkdir(exist_ok=False);started=time.time()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    (OUT/'gpu_before.txt').write_text(rows)
    idle=[r.split(',')[0].strip() for r in rows.splitlines() if int(r.split(',')[2])==0 and int(r.split(',')[3])==0]
    if not idle:save('blocked.json',dict(reason='no idle GPU'));return
    gpu=idle[0]
    old=json.loads((ROOT/'results/terminal_structure_real_20261109T233600/protocol.json').read_text())
    bindings=old['bindings'].copy()
    prior=json.loads((ROOT/'results/one_step_vs_terminal_plan_20261111/bindings.json').read_text())
    bindings.update({p:h for p,h in prior.items() if '/terminal_structure_real_' in p})
    for ds in old['models']:
        for ss in old['sample_seeds']:
            p=ROOT/'results/terminal_structure_real_20261109T233600'/ds/str(ss)/'graph.npz'
            expected=json.loads(p.with_name('summary.json').read_text())['graph_sha256']
            assert sha(p)==expected
            bindings[str(p)]=expected
    for p in [Path(__file__),Path(__file__).with_name('gpu_preflight.py'),
              ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json',
              ROOT/'research_branches/typed_heat_audit_20261111/run.py',
              ROOT/'research_branches/typed_heat_degree_audit_20261111/run.py']:
        bindings[str(p)]=sha(p)
    for p,h in bindings.items():assert sha(p)==h, 'changed asset: '+p
    save('protocol.json',dict(start_epoch=started,stop_epoch=started+900,wall_minutes=15,GPU_hours=.25,
        gpu=gpu,bindings=bindings,forward_limit=64,VJP_limit=192,updates=0,new_reverse_chains=0,
        arms=['S1 single draw/no baseline','S16 per-draw relative LOO reward','terminal eager clean-head/4-terminal relative reward'],
        score='frozen typed joint heat-degree MMD + per-family squared count error, unit coefficients diagnostic only',
        mode='historical strict deterministic eval, TF32 false; no new restoration tolerance',
        selection='all16 draws retained; no best-only',
        limitation='saved generation-state engineering panel, not forward-noise training comparison or effect evidence'))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
    with (OUT/'worker.log').open('w') as log:
        allocation=time.time()
        p=subprocess.Popen([PY,str(Path(__file__).with_name('gpu_preflight.py')),str(OUT)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save('launch.json',dict(pid=p.pid,gpu=gpu,allocation_epoch=allocation))
        while p.poll() is None:
            now=time.time()
            save('watchdog.json',dict(at_epoch=now,GPU_seconds=now-allocation,pid=p.pid))
            if now>=started+895 or now-allocation>=895:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(1)
        p.wait();ended=time.time()
    save('release.json',dict(returncode=p.returncode,wall_seconds=ended-started,GPU_seconds=ended-allocation,GPU_hours=(ended-allocation)/3600,
        gpu=gpu,all_owned_processes_exited=True))
    (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
    print((OUT/'release.json').read_text(),flush=True)
if __name__=='__main__':main()
