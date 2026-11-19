"""Register environment-matched controls before any new outcome is available."""
import json,hashlib,shutil,tarfile,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
PARENT=ROOT/'results/joint_output_formal_20261108T021741'
OUT=ROOT/'results/joint_output_formal_controls_20261108T025000'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    if (OUT/'source_binding.json').exists():raise RuntimeError('already frozen')
    for d in ['engineering','prepared','leases','logs','run','evaluation']:(OUT/d).mkdir(parents=True,exist_ok=True)
    ledger=read(PARENT/'ledger.json');ledger.update(shared_lease_roots=[str(PARENT),str(OUT)],same_ticket=True)
    save(OUT/'ledger.json',ledger)
    save(OUT/'approval.json',dict(authorized=True,launcher_pid=None,scope='8 same-environment O/A CE controls;3000updates+32chains; no new candidate',
        reason='Historical O/A strict deterministic sparse_block environment differs from PB/new arms ordinary dihug environment',
        user_instruction='autonomous8h complete output factorial; resources authorized; no restart of clock'))
    tasks=[]
    for arch in ['O','A']:
        for i,t in enumerate(read(PARENT/'tasks.json')[:4]):
            base=f"{t['dataset']}_{t['train_seed']}_{arch}_late"
            original=Path(t['initial']).parent.parent/base/'initial.pt'
            # Config is identical across O/A; original identity is checked again by runtime.load.
            tasks.append(dict(t,task_id=f"{t['dataset']}_{t['train_seed']}_{arch}_CE",baseline_task=base,
                arm=arch+'_CE',architecture=arch,objective_kind='CE',gpu=4+i,queue_order=len(tasks),
                initial=str(original),initial_sha256=sha(original),engineering=i in [0,2]))
    save(OUT/'tasks.json',tasks)
    jobs=read(PARENT/'evaluation_jobs.json')
    for j in jobs:
        if j['arm'] in ['O','A']:
            t=next(t for t in tasks if t['dataset']==j['dataset'] and t['train_seed']==j['train_seed'] and t['architecture']==j['arm'])
            f=OUT/'run'/t['task_id'];j.update(task_id=t['task_id'],base_task=t['task_id'],folder=str(f/'ema_0.99'/f"sample_{j['sample_seed']}"),
                readout=str(f/'readout_ema_0.99.pt'),configuration=t['arm']+'_late',reused=False)
            j.pop('metric_dir',None)
        elif not j['reused']:
            j['reused']=True;j['metric_dir']=str(PARENT/'evaluation'/j['task_id']/'u3000'/str(j['sample_seed']))
    save(OUT/'evaluation_jobs.json',jobs)
    for p in (PARENT/'prepared').glob('*.json'):shutil.copy2(p,OUT/'prepared'/p.name)
    binding=read(PARENT/'source_binding.json')
    for p,h in binding.items():assert sha(p)==h,p
    for p in SRC.rglob('*.py'):binding[str(p)]=sha(p)
    save(OUT/'source_binding.json',binding)
    save(OUT/'source_version.json',dict(parent_source_manifest=str(PARENT/'source_binding.json'),parent_sha256=sha(PARENT/'source_binding.json'),
        manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated same-environment CE controls; production CE update delegated unchanged'))
    env=[]
    for label,p in [('old_O',Path(tasks[0]['initial']).parent/'environment.json'),
        ('old_A',Path(tasks[4]['initial']).parent/'environment.json'),
        ('PB',ROOT/'results/pb_degree_formal_20261106T170600/run/IMDB_2026110411_A_late/environment.json'),
        ('current',PARENT/'run/IMDB_2026110411_O_PB/environment.json')]:
        env.append(dict(label=label,path=str(p),sha256=sha(p),environment=read(p)))
    save(OUT/'environment_discrepancy.json',dict(discovered_before_new_generation=True,records=env,
        implication='Historical O/A vs output arms is not strictly runtime matched; retain historical table and replace primary CE controls with new same-environment runs. No claim environment caused prior quality differences.'))
    # Cost bound from >800 concurrent full updates per active model, not a new scientific trial.
    p95={}
    for p in (PARENT/'run').glob('*/training.jsonl'):
        values=sorted(json.loads(s)['total_update_seconds'] for s in p.read_text().splitlines()[5:]);ds=p.parent.name.split('_')[0]
        p95[ds]=max(p95.get(ds,0),values[int(.95*len(values))])
    old=read(PARENT/'budget.json');existing=[];added=[]
    for t in old['tasks']:
        ds=t['task'].split('_')[0];existing.append(dict(task=t['task'],device_minutes=max(p95[ds],t['measured_boundary_update_seconds'])*1.25*50+t['chains_minutes']+10))
    for t in tasks:
        ds=t['dataset'];mins=p95[ds]*1.25*50+max(x['chains_minutes'] for x in old['tasks'] if x['task'].startswith(ds))+10
        added.append(dict(task=t['task_id'],gpu=t['gpu'],device_minutes=mins))
    total=(sum(x['device_minutes'] for x in existing+added))/60+1.1
    # Count queues from original launch; engineering/load slack included. New controls only on4-7 after original owners release.
    queue=max(old['tasks'][i+4]['device_minutes']+sum(x['device_minutes'] for x in added if x['gpu']==i+4) for i in range(4))+5
    start=read(PARENT/'formal1_started.json')['epoch'];finish=start+queue*60
    fits=total<ledger['GPU_hours_limit'] and finish<ledger['science_stop_epoch']-120
    save(OUT/'budget.json',dict(fits=fits,epoch=time.time(),queue_minutes=queue,estimated_finish_epoch=finish,combined_GPU_hours=total,
        current_p95_full_update_seconds=p95,existing_task_bounds=existing,new_task_bounds=added,
        margin='25% on concurrent p95 update (using slower auxiliary arms to bound CE), original full-chain margin,10min/card I/O,1.1GPUh global reserve',
        original_clock_preserved=True,shared_gpu_hour_ceiling=ledger['GPU_hours_limit']))
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in binding:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    print(json.dumps(dict(out=str(OUT),fits=fits,combined_GPU_hours=total,queue_minutes=queue)))
if __name__=='__main__':main()
