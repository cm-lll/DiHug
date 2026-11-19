"""Authorized 120-minute / 8-card-hour queue and independent watchdog."""
import os,sys,json,time,signal,subprocess,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
PLAN=ROOT/'results/hetero_micro_macro_effect_plan_20261116'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
START=1794789082.0
TRAIN=HERE/'effect_worker.py';GEN=HERE/'effect_generation.py';ANALYSIS=HERE/'effect_analysis.py'
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
            if (str(TRAIN).encode() in cmd or str(GEN).encode() in cmd) and os.getpgid(r['pid'])==r['pid']:os.killpg(r['pid'],sig)
        except (ProcessLookupError,FileNotFoundError):pass
def watchdog():
    p=json.loads((OUT/'protocol.json').read_text())
    while not (OUT/'gpu_finished.json').exists():
        try:rows=json.loads((OUT/'allocations.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
        active=[r for r in rows if not r.get('end_epoch')];now=time.time()
        if now>=p['science_stop_epoch']-15 or used(rows)>=p['GPU_seconds_limit']-max(1,len(active))*15 or any(now>=r['stage_deadline'] for r in active):
            write(OUT/'watchdog_stop.json',dict(epoch=now,GPU_seconds=used(rows),reason='wall/card/stage limit'))
            (OUT/'STOP').write_text('budget boundary\n');stop_owned(rows,signal.SIGTERM);time.sleep(8);stop_owned(rows,signal.SIGKILL);return
        time.sleep(1)
def prepare():
    OUT.mkdir(exist_ok=False);p=json.loads((PLAN/'protocol.json').read_text());bindings=json.loads((PLAN/'asset_bindings.json').read_text())
    for f,h in bindings.items():assert sha(f)==h,f
    for f in [TRAIN,GEN,ANALYSIS,HERE/'effect_hard_score.py',HERE/'gradient_scale.py',HERE/'objective.py',Path(__file__),
              ROOT/'research_branches/new_datasets_20261112/metrics.py',ROOT/'research_branches/joint_transport_ablation_20261113/reliability_analysis.py']:bindings[str(f)]=sha(f)
    p.update(status='authorized_execution',requires_new_unified_budget=False,start_epoch=START,science_stop_epoch=START+6000,delivery_epoch=START+7200,
        GPU_seconds_limit=28800,runtime_source_bindings={f:h for f,h in bindings.items() if f.endswith('.py')},checkpoints=[],
        calibration_stage_seconds_limit=900,training_worker_seconds_limit=1800)
    for t in p['tasks']:t['readout']='ema'
    write(OUT/'protocol.json',p);write(OUT/'asset_recheck.json',dict(bindings=bindings,epoch=time.time(),unchanged=True))
    write(OUT/'allocations.json',[])
    print(json.dumps(dict(output=str(OUT),start=START,science_stop=START+6000,delivery=START+7200)))
def main():
    if (OUT/'supervisor.json').exists():raise RuntimeError('refuse duplicate supervisor')
    p=json.loads((OUT/'protocol.json').read_text());assert time.time()<p['science_stop_epoch']
    write(OUT/'supervisor.json',dict(pid=os.getpid(),epoch=time.time()))
    rows=[];active={};wd=subprocess.Popen([PY,__file__,'watchdog'],stdout=(OUT/'watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    def resource():
        txt=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=25)
        return txt,{int(x.split(', ')[0]):x.split(', ') for x in txt.strip().splitlines()}
    def launch(script,args,g,tag,stage_seconds):
        if (OUT/'STOP').exists() or time.time()>=p['science_stop_epoch']-15:raise TimeoutError('launch cutoff')
        txt,cards=resource();assert g in cards and int(cards[g][2])<100,f'GPU{g} busy'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        tick=time.time();log=(OUT/(tag+'.log')).open('w')
        proc=subprocess.Popen([PY,str(script),str(OUT),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(tag=tag,gpu=g,uuid=cards[g][1],pid=proc.pid,start_epoch=tick,end_epoch=None,stage_deadline=min(tick+stage_seconds,p['science_stop_epoch']-15))
        rows.append(row);active[proc.pid]=(proc,row,log);write(OUT/'allocations.json',rows)
    def drain():
        while active:
            for pid,(proc,row,log) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                row.update(end_epoch=time.time(),returncode=rc);log.close();del active[pid];write(OUT/'allocations.json',rows)
            if active:time.sleep(1)
    def require_success():
        if any(r.get('returncode',0)!=0 for r in rows):raise RuntimeError('worker failed; preserve partial work, do not launch dependent stage')
    try:
        txt,cards=resource();write(OUT/'resources.json',dict(epoch=time.time(),GPUs=txt))
        assert len(cards)>=8 and all(int(cards[g][2])<100 for g in range(8)),'full queue requires eight initially free devices'
        launch(TRAIN,['calibrate'],0,'calibration',900);drain();require_success()
        coeff=json.loads((OUT/'gradient_coefficients.json').read_text());assert coeff['network_forwards']==24 and coeff['network_VJPs']==96
        assets=json.loads((OUT/'asset_recheck.json').read_text());assets['bindings'][str(OUT/'gradient_coefficients.json')]=sha(OUT/'gradient_coefficients.json');write(OUT/'asset_recheck.json',assets)
        generation=1.15*p['budget']['generation_body_GPU_hours']*3600
        train=1.25*p['budget']['training_body_GPU_hours']*3600
        remaining=generation+train+2700
        max_chain=max(x['seconds_max'] for x in p['budget']['historical_chain_costs'])*1.1
        wall=128*max(x['candidate_seconds'] for x in p['budget']['measured_update_costs'])*1.25+4*max_chain+600
        passed=used(rows)+remaining<p['GPU_seconds_limit'] and time.time()+wall<p['science_stop_epoch']
        write(OUT/'training_cost_gate.json',dict(passed=passed,GPU_hours_used=used(rows)/3600,remaining_GPU_hours=remaining/3600,remaining_wall_minutes=wall/60,whole_chains_extrapolated=True))
        if not passed:raise RuntimeError('complete panel cannot fit remaining budget; no formal arm started')
        for i in range(4):
            for j,arm in enumerate(['CE','calibrated_micro_macro']):launch(TRAIN,['train',i,arm],2*i+j,f'train_{i}_{arm}',1800)
        write(OUT/'training_launch.json',dict(epoch=time.time(),jobs=8,updates_per_model=128,quality_gate=False,allocations=rows))
        drain();require_success()
        for i,row in enumerate(p['parents']):
            base=None
            for arm in ['CE','calibrated_micro_macro']:
                folder=OUT/row['id']/arm;r=json.loads((folder/'completion.json').read_text());assert r['status']=='complete' and r['updates']==128
                inputs=json.loads((folder/'inputs.json').read_text())
                if base is None:base=inputs
                else:assert inputs==base,'training input pairing failed'
                assert sha(r['checkpoint'])==r['checkpoint_sha256']
                cp=dict(model=row['id'],arm=arm,readout='ema',checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],model_hash=r['model_hash'],updates=128)
                p['checkpoints'].append(cp)
                for t in p['tasks']:
                    if t['model']==row['id'] and t['arm']==arm:t.update(checkpoint=cp['checkpoint'],model_hash=cp['model_hash'])
        write(OUT/'protocol.json',p)
        # First ten steps are part of the fixed 32-chain panel, never extra draws.
        for i,g in [(0,0),(2,4)]:launch(GEN,[i,'ema','preflight','calibrated_micro_macro'],g,f'sample_preflight_{i}',300)
        drain();require_success();cal={}
        for i in (0,2):
            row=p['parents'][i];folder=OUT/row['id']/'calibrated_micro_macro'/'ema';d=json.loads((folder/'calibration.json').read_text())
            logs=[json.loads(x) for x in next(folder.glob('calibrated_micro_macro_seed*/steps.jsonl')).read_text().splitlines()]
            assert len(logs)==10;cal[row['dataset']]=dict(**d,steady=sum(x['step_seconds'] for x in logs[2:])/8)
        estimates=[dict(model=x['model'],seconds=max(x['seconds_mean']*1.15,x['seconds_max'],cal[x['dataset']]['steady']*1100)) for x in p['budget']['historical_chain_costs']]
        remaining=sum(x['seconds']*8 for x in estimates)+600;wall=max(x['seconds']*4 for x in estimates)+180
        passed=used(rows)+remaining<p['GPU_seconds_limit'] and time.time()+wall<p['science_stop_epoch']
        write(OUT/'generation_cost_gate.json',dict(passed=passed,estimates=estimates,calibration=cal,GPU_hours_used=used(rows)/3600,remaining_GPU_hours=remaining/3600,remaining_wall_minutes=wall/60))
        if not passed:raise RuntimeError('full generation panel cannot fit; preserve trained states and partial chains')
        for i in range(4):
            for j,arm in enumerate(['CE','calibrated_micro_macro']):launch(GEN,[i,'ema','run',arm],2*i+j,f'generate_{i}_{arm}',6000)
        write(OUT/'generation_launch.json',dict(epoch=time.time(),new_chains=32,reused_graphs=16,quality_gate=False,allocations=rows))
        drain();require_success();status='complete'
    except BaseException:
        write(OUT/'supervisor_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()));stop_owned(rows,signal.SIGTERM)
        deadline=time.time()+10
        while active and time.time()<deadline:
            if all(proc.poll() is not None for proc,r,l in active.values()):break
            time.sleep(.5)
        stop_owned(rows,signal.SIGKILL);drain();status='stopped'
    finally:
        write(OUT/'gpu_finished.json',dict(epoch=time.time(),GPU_hours=used(rows)/3600,status=locals().get('status','failed')))
        wd.wait(timeout=10)
        try:txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=25)
        except Exception as e:txt='unavailable: '+repr(e)
        write(OUT/'release.json',dict(epoch=time.time(),all_owned_processes_exited=not active,GPU_hours=used(rows)/3600,host_compute_processes=txt))
    try:
        rc=subprocess.run([PY,str(ANALYSIS),str(OUT)],stdout=(OUT/'analysis.log').open('w'),stderr=subprocess.STDOUT,timeout=max(1,p['delivery_epoch']-time.time()-10)).returncode
        if rc:raise RuntimeError(f'CPU evaluation exit {rc}')
    except BaseException:write(OUT/'delivery_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
if __name__=='__main__':{'prepare':prepare,'run':main,'watchdog':watchdog}[sys.argv[1]]()
