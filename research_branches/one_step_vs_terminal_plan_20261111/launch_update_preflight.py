"""Two owned workers,30 minute wall/1 GPU-hour ceiling,25 minute science."""
import os,time,json,sys,hashlib,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/three_arm_update_preflight_20261111'
PY=__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def main():
    OUT.mkdir(exist_ok=False);start=time.time()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    (OUT/'gpu_before.txt').write_text(rows)
    free=[r.split(',')[0].strip() for r in rows.splitlines() if int(r.split(',')[2])==0 and int(r.split(',')[3])==0]
    if len(free)<2:save('blocked.json',dict(reason='need two idle GPUs'));return
    old=json.loads((ROOT/'results/one_step_terminal_gpu_preflight_20261111/protocol.json').read_text())
    bindings=old['bindings'].copy()
    for p in [Path(__file__),Path(__file__).with_name('update_preflight.py'),
        ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/ema.py',
        ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/integration.py',
        ROOT/'research_branches/cn_multiseed_20261104/research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/boundary_transactions.py']:
        bindings[str(p)]=sha(p)
    for p,h in bindings.items():assert sha(p)==h,p
    save('protocol.json',dict(start_epoch=start,science_stop_epoch=start+1500,delivery_epoch=start+1800,GPU_hours=1,max_GPUs=2,
        bindings=bindings,datasets=['IMDB','PubMed'],train_seeds=[2026110411,2026110421],
        models='same pretrained EMA3000 starts; fresh AdamW and EMA for all arms; remaining optimizer fields from parent config',
        optimizer_lr=1e-5,EMA=.99,trainable='all original A parameters, no new module',loss_scale=.01,gradient_clip=1,
        reward='negative (typed joint heat-degree MMD + per-family count risk), no standardization/no reward clipping',
        estimators=dict(S1='single sample no baseline',S16='all16 distinct per-sample LOO advantages',
            T_eager='four fixed parent trajectories,8times each,uniform selected-time average,LOO terminal advantage; biased eager surrogate'),
        scope='2temporary updates per arm x3arms x2datasets=12,then discard temporary weights;2new unchanged-parent T1000 baseline chains',
        terminal_second_update='fixed old batch restoration test only, NOT on-policy learning and excluded from efficacy claims',
        scientific_effect_claim=False,first_sample_seed=2026111341,main_input_root=2026111401,hard_root=2026111404,
        gates='exact loaded model/AdamW/EMA/RNG/counter; identical next input; finite gradients/displacements; validation restores state; unchanged support; measured complete-chain fit',
        not_gates='no short-effect improvement requirement; no redefinition of old floating recomputation tolerance'))
    procs=[]
    for gpu,ds in zip(free[:2],['IMDB','PubMed']):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;log=(OUT/(ds+'.log')).open('w');allocation=time.time()
        p=subprocess.Popen([PY,str(Path(__file__).with_name('update_preflight.py')),str(OUT),ds],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        procs.append(dict(proc=p,gpu=gpu,dataset=ds,start=allocation,end=None,log=log))
    save('launch.json',[dict(pid=x['proc'].pid,gpu=x['gpu'],dataset=x['dataset'],start_epoch=x['start']) for x in procs])
    stopped=False
    while True:
        now=time.time()
        for x in procs:
            if x['end'] is None and x['proc'].poll() is not None:x['end']=now
        hours=sum((x['end'] or now)-x['start'] for x in procs)/3600
        save('watchdog.json',dict(at_epoch=now,GPU_hours=hours,processes=[dict(pid=x['proc'].pid,dataset=x['dataset'],returncode=x['proc'].poll()) for x in procs]))
        if all(x['end'] is not None for x in procs):break
        if not stopped and (now>=start+1495 or hours>=.998):
            stopped=True
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGTERM)
            time.sleep(3)
            for x in procs:
                if x['proc'].poll() is None:os.killpg(x['proc'].pid,signal.SIGKILL)
        time.sleep(1)
    for x in procs:x['log'].close()
    save('release.json',dict(wall_seconds=time.time()-start,GPU_hours=hours,stopped_at_limit=stopped,
        processes=[dict(dataset=x['dataset'],gpu=x['gpu'],returncode=x['proc'].returncode,seconds=x['end']-x['start']) for x in procs]))
    (OUT/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))
    print((OUT/'release.json').read_text(),flush=True)
if __name__=='__main__':main()
