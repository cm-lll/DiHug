"""Bind one authorized 8-hour ticket. CPU assets only, no new scientific model calls."""
import datetime,hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
OUT=ROOT/'results/joint_output_formal_20261108T021741'
BASE=ROOT/'results/cn_multiseed_20261104T042546';PB=ROOT/'results/pb_degree_formal_20261106T170600'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    if (OUT/'source_binding.json').exists() or list((OUT/'leases').glob('*.json')):raise RuntimeError('already frozen or computing; do not reset clock or overwrite')
    if (OUT/'tasks.json').exists():
        save(OUT/'preparation_failure1.json',dict(error='PB evaluation directory glob also selected reused A metrics',fix='select exact PB metric folder; no GPU or model computation occurred',original_start_preserved=True))
    for d in ['engineering','prepared','leases','logs','run','evaluation']:(OUT/d).mkdir(parents=True,exist_ok=True)
    start=datetime.datetime.fromisoformat('2026-11-08T02:17:41+08:00').timestamp()
    save(OUT/'ledger.json',dict(start_epoch=start,science_stop_epoch=start+450*60,deadline_epoch=start+480*60,
        GPU_hours_limit=32,engineering_GPU_hours_limit=1,ordinary_recompute_tolerance_passed=False,
        limit_note='8h user window; self-imposed32 GPUh ceiling; max8 free2080Ti; ownership ledger includes waits'))
    save(OUT/'approval.json',dict(authorized=True,scope='one frozen joint-output candidate and complete historical PB factorial;12 new models3000+48 T1000;12 reused models48graphs',launcher_pid=None,automatic_followup=False,
        user_instruction='8h autonomous output correction; include O+PB; no questions; no guarantee of positive outcomes'))
    old=read(PB/'tasks.json');tasks=[];assets={};jobs=[]
    for kind,arch in [('PB','O'),('JK','A'),('JK','O')]:
        for i,t in enumerate(old):
            ds,seed=t['dataset'],t['train_seed'];base=f'{ds}_{seed}_{arch}_late';initial=BASE/'run'/base/'initial.pt'
            h=sha(initial);z=torch.load(initial,map_location='cpu')
            assert z['updates']==0 and not z['optimizer']['state'] and z['ema']['updates']==0
            assert z['binding']['arm']==arch and z['binding']['dataset']==ds and z['binding']['train_seed']==seed
            assert z['model']['joint_head.0.weight'].shape==(128,776 if arch=='A' else 739)
            assets[str(initial)]=h
            idx=len(tasks);arm=arch+'_'+kind
            tasks.append(dict(task_id=f'{ds}_{seed}_{arm}',baseline_task=base,dataset=ds,train_seed=seed,arm=arm,architecture=arch,objective_kind=kind,
                gpu=idx%8,queue_order=idx,config=z['binding']['config'],initial=str(initial),initial_sha256=h,
                engineering=(seed in [2026110411,2026110421])))
    save(OUT/'tasks.json',tasks)
    # Reuse ordinary O, ordinary A and historical A+PB without selecting graphs.
    for t in old:
        ds,seed=t['dataset'],t['train_seed']
        for arm,root,arch in [('O',BASE,'O'),('A',BASE,'A'),('A_PB',PB,'A')]:
            tid=f'{ds}_{seed}_{arch}_late';folder=root/'run'/tid
            for ss in [2026102821,2026102822,2026110231,2026110232]:
                sample=folder/'ema_0.99'/f'sample_{ss}';b=read(sample/'binding.json')
                assert b['dataset']==ds and b['train_seed']==seed and b['updates']==3000 and b['sample_seed']==ss
                for p,expected in [(sample/'graph.npz',b['graph_sha256']),(folder/'readout_ema_0.99.pt',b['model_sha256'])]:
                    assert sha(p)==expected;assets[str(p)]=expected
                # Existing full metric records located by their exact graph hash.
                metric_files=list((root/'evaluation').glob(f'{tid}_PB/u3000/{ss}/original.json')) if arm=='A_PB' else list((root/'evaluation').glob(f'{tid}/u3000/{ss}/original.json'))
                if not metric_files:metric_files=list((root/'evaluation').glob(f'{tid}*/{ss}/original.json'))
                if len(metric_files)!=1:raise ValueError(('metric path ambiguity',tid,ss,metric_files))
                original=read(metric_files[0]);assert original['graph_binding']['graph_sha256']==b['graph_sha256']
                jobs.append(dict(dataset=ds,train_seed=seed,arm=arm,task_id=tid+'_'+arm,base_task=tid,updates=3000,sample_seed=ss,
                    folder=str(sample),readout=str(folder/'readout_ema_0.99.pt'),configuration=b['configuration'],reused=True,metric_dir=str(metric_files[0].parent)))
    for t in tasks:
        for ss in [2026102821,2026102822,2026110231,2026110232]:
            folder=OUT/'run'/t['task_id']
            jobs.append(dict(dataset=t['dataset'],train_seed=t['train_seed'],arm=t['arm'],task_id=t['task_id'],base_task=t['task_id'],updates=3000,sample_seed=ss,
                folder=str(folder/'ema_0.99'/f'sample_{ss}'),readout=str(folder/'readout_ema_0.99.pt'),configuration=t['arm']+'_late',reused=False))
    save(OUT/'evaluation_jobs.json',jobs)
    for p in (PB/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name);assets[str(p)]=sha(p)
    # Actual parent science files are frozen by historical PB manifest.
    sources=read(PB/'source_binding.json')
    for p,h in sources.items():assert sha(p)==h,p
    for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources);save(OUT/'assets.json',assets)
    save(OUT/'source_version.json',dict(parent_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated source SHA manifest; additions not claimed to be in parent commit'))
    save(OUT/'asset_audit.json',dict(passed=True,new_models=12,new_chains=48,reused_models=12,reused_graphs=48,disk_free_GiB=shutil.disk_usage(ROOT).free/2**30,
        ordinary_recompute_tolerance_passed=False,complete_load_exact_required=True,checkpoint_step=3000,normal1=True,EMA=.99,
        limitation='development seeds, not independent confirmatory panel; old O/A parameter counts differ; no bitwise GPU recomputation claim'))
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    print(json.dumps(dict(out=str(OUT),tasks=len(tasks),graphs=len(jobs),elapsed_minutes=(time.time()-start)/60)))
if __name__=='__main__':main()
