"""Owned process queue, independently enforced wall/allocation limits, CPU delivery."""
import os,sys,time,json,signal,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/analytic_joint_generation_run_20261113T225800'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
WORKER=Path(__file__).with_name('generation_worker.py')
ANALYSIS=Path(__file__).with_name('generation_analysis.py')
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def used(rows):return sum((r.get('end_epoch') or time.time())-r['start_epoch'] for r in rows)
def stop_owned(rows,sig):
    for r in rows:
        if r.get('end_epoch'):continue
        try:
            cmd=Path(f'/proc/{r["pid"]}/cmdline').read_bytes()
            if str(WORKER).encode() in cmd and os.getpgid(r['pid'])==r['pid']:os.killpg(r['pid'],sig)
        except (ProcessLookupError,FileNotFoundError):pass
def watchdog():
    p=json.loads((OUT/'protocol.json').read_text())
    while not (OUT/'gpu_finished.json').exists():
        try:rows=json.loads((OUT/'allocations.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
        # Leave 15 seconds per active card for safe shutdown, accounted as occupation.
        active=sum(not r.get('end_epoch') for r in rows)
        if time.time()>=p['science_stop_epoch']-15 or used(rows)>=p['GPU_seconds_limit']-max(1,active)*15:
            write(OUT/'watchdog_stop.json',dict(epoch=time.time(),GPU_seconds=used(rows),reason='wall or allocation limit'))
            (OUT/'STOP').write_text('budget boundary\n');stop_owned(rows,signal.SIGTERM);time.sleep(8)
            stop_owned(rows,signal.SIGKILL);return
        time.sleep(1)
def main():
    if (OUT/'supervisor.json').exists():raise RuntimeError('refuse duplicate supervisor')
    write(OUT/'supervisor.json',dict(pid=os.getpid(),epoch=time.time()))
    p=json.loads((OUT/'protocol.json').read_text())
    plan=ROOT/'results/analytic_joint_generation_plan_20261113'
    bindings=json.loads((plan/'asset_bindings.json').read_text())
    for path,h in bindings.items():assert sha(path)==h,path
    for path in (WORKER,ANALYSIS,Path(__file__)):bindings[str(path)]=sha(path)
    # Analysis dependencies: bind existing metrics and their frozen dependencies too.
    for path in [ROOT/'research_branches/new_datasets_20261112/metrics.py',
                 ROOT/'research_branches/joint_transport_ablation_20261113/reliability_analysis.py']:
        bindings[str(path)]=sha(path)
    p.update(status='authorized_execution',runtime_source_bindings={a:h for a,h in bindings.items() if a.endswith('.py')},
             execution_source_bindings={str(a):sha(a) for a in (WORKER,ANALYSIS,Path(__file__))})
    write(OUT/'protocol.json',p);write(OUT/'asset_recheck.json',dict(epoch=time.time(),all_unchanged=True,bindings=bindings))
    rows=[];active={};write(OUT/'allocations.json',rows)
    wd=subprocess.Popen([PY,str(Path(__file__)),'watchdog'],start_new_session=True,stdout=(OUT/'watchdog.log').open('w'),stderr=subprocess.STDOUT)
    def free(g):
        v=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(v.strip())>100:raise RuntimeError(f'GPU{g} unavailable: {v}')
    def launch(mi,readout,mode,g):
        if (OUT/'STOP').exists() or time.time()>=p['science_stop_epoch']:raise TimeoutError('launch cutoff')
        free(g);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=(OUT/f'worker_{mi}_{readout}_{mode}.log').open('w');tick=time.time()
        proc=subprocess.Popen([PY,str(WORKER),str(OUT),str(mi),readout,mode],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        r=dict(model=mi,readout=readout,mode=mode,gpu=g,pid=proc.pid,start_epoch=tick,end_epoch=None)
        rows.append(r);active[proc.pid]=(proc,r,log);write(OUT/'allocations.json',rows)
    def drain():
        while active:
            for pid,(proc,r,log) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                r.update(end_epoch=time.time(),returncode=rc);log.close();del active[pid];write(OUT/'allocations.json',rows)
            if active:time.sleep(1)
    try:
        for g in range(8):free(g)
        launch(0,'raw','preflight',0);launch(2,'raw','preflight',4);drain()
        if any(r['returncode']!=0 for r in rows):raise RuntimeError('preflight failed; full queue not launched')
        calibration={}
        for mi in (0,2):
            row=p['parents'][mi];d=OUT/row['id']/'raw';cal=json.loads((d/'calibration.json').read_text())
            first=list(d.glob('bias_seed*/steps.jsonl'));assert len(first)==1
            records=[json.loads(x) for x in first[0].read_text().splitlines()];assert len(records)==10
            steady=sum(x['step_seconds'] for x in records[2:])/8
            calibration[row['dataset']]=dict(**cal,steady_last8_seconds_per_step=steady)
        estimates=[]
        for history in p['budgets']['history']:
            # Fixed rule: warmup already paid, max historical+25% and actual steady+10%.
            chain=max(history['seconds_mean']*1.25,calibration[history['dataset']]['steady_last8_seconds_per_step']*1000*1.10)
            estimates.append(dict(model=history['model'],seconds_per_chain=chain))
        remaining=sum(r['seconds_per_chain']*16 for r in estimates)+240
        wall=max(r['seconds_per_chain']*8 for r in estimates)+120
        passed=used(rows)+remaining<p['GPU_seconds_limit'] and time.time()+wall<p['science_stop_epoch']
        write(OUT/'cost_gate.json',dict(passed=passed,epoch=time.time(),calibration=calibration,estimates=estimates,
            GPU_hours_used=used(rows)/3600,predicted_remaining_GPU_hours=remaining/3600,
            predicted_remaining_wall_minutes=wall/60,whole_chain_still_extrapolated=True))
        if not passed:raise RuntimeError('remaining complete panel does not fit budget; no partial seed selection')
        for mi in range(4):
            for ri,readout in enumerate(('raw','ema')):launch(mi,readout,'run',2*mi+ri)
        write(OUT/'launch_acceptance.json',dict(epoch=time.time(),budget_passed=True,preflight_steps=20,
            new_chains_planned=64,reused_graphs=16,allocations=rows,CUDA_bitwise_recompute_certified=False))
        drain();status='complete' if all(r['returncode']==0 for r in rows) else 'partial_failure'
    except BaseException:
        write(OUT/'supervisor_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        stop_owned(rows,signal.SIGTERM)
        # A bounded forced exit is restricted to our registered process groups.
        deadline=time.time()+10
        while active and time.time()<deadline:
            if all(proc.poll() is not None for proc,r,l in active.values()):break
            time.sleep(.5)
        stop_owned(rows,signal.SIGKILL);drain();status='stopped'
    finally:
        write(OUT/'gpu_finished.json',dict(epoch=time.time(),GPU_hours=used(rows)/3600,status=locals().get('status','failed')))
        wd.wait(timeout=10)
        txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=20)
        write(OUT/'release.json',dict(epoch=time.time(),all_owned_processes_exited=not active,GPU_hours=used(rows)/3600,host_compute_processes=txt))
    timeout=max(1,p['delivery_epoch']-time.time()-15)
    try:
        rc=subprocess.run([PY,str(ANALYSIS),str(OUT)],stdout=(OUT/'analysis.log').open('w'),stderr=subprocess.STDOUT,timeout=timeout).returncode
        if rc:raise RuntimeError(f'CPU analysis exit {rc}')
    except BaseException:write(OUT/'delivery_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
if __name__=='__main__':watchdog() if len(sys.argv)>1 and sys.argv[1]=='watchdog' else main()
