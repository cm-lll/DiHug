"""Generate preparation evidence only; never launches science."""
import pathlib,sys,json,hashlib,shutil,difflib,math,time,os
OUT=pathlib.Path(__file__).resolve().parent
REPO=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/typed_structure_early_cn_expansion_review')
ROOT=REPO/'experiments/typed_structure_early_cn_expansion_v1';BASE=ROOT.parent/'typed_structure_early_cn_v1'
sys.path.insert(0,str(ROOT));from runtime_contract import jobs,source_binding,PROTOCOL
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
ev=ROOT/'evidence';ev.mkdir(exist_ok=True)
for d in ('IMDB','PubMed'):
    r=read(OUT/(d+'_D/summary.json'));assert r['status']=='passed' and r['optimizer_step_calls']==3
    assert all(x['finite'] for x in r['inference']) and all(r['loaded_exact'].values())
    shutil.copy2(OUT/(d+'_D/summary.json'),ev/(d+'_D_engineering.json'))
    shutil.copy2(OUT/(d+'_D/initial_binding.json'),ev/(d+'_D_initial_binding.json'))
for file in ['ledger.json','seed_audit.json','old_seed_DE_pairing.json','protected_snapshot.json','passive_handle_owners.json']:
    if (OUT/file).exists():shutil.copy2(OUT/file,ev/file)
for file in ['receipt.json','literal_original_constructor_audit.json']:
    shutil.copy2(OUT/'new_seed_pairing'/file,ev/('pairing_'+file))
for file in ['cpu_tests1.log','cpu_tests2.log','pairing.log']:
    shutil.copy2(OUT/'logs'/file,ev/file)
artifacts=[]
for folder in ['IMDB_D','PubMed_D','new_seed_pairing']:
    for p in sorted((OUT/folder).rglob('*')):
        if p.is_file():artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),classification='engineering only; never a formal training checkpoint'))
save(ev/'artifact_index.json',artifacts)
shared={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or any(x in p.parts for x in ('__pycache__','.pytest_cache','evidence')):continue
    rel=str(p.relative_to(ROOT));q=BASE/rel
    if p.suffix not in ('.py','.npz','.json') or rel in ('source_lineage.json','source_manifest.json','cost.json'):continue
    shared[rel]=dict(baseline_sha256=sha(q) if q.exists() else None,candidate_sha256=sha(p),unchanged=q.exists() and sha(p)==sha(q))
save(ROOT/'source_lineage.json',dict(reviewed_early_commit='dbe96f519443e8f8937d43cd191803658bf83464',original_source_commit='472e533fd28cfc067702b1b75082ac5105140a2d',original_result_commit='aa4227b9727ca6853ba7e26412ce6d438a0fa971',files=shared,protected_source_modified=False))
changed_package=[p for p,x in shared.items() if p.startswith('package/') and not x['unchanged']]
assert changed_package==['package/early_cn.py'],changed_package
assert all(shared[p]['unchanged'] for p in ['run_worker.py','ema.py','integration.py','evaluation_fields.py','package/normal2.py'])
patch=[]
for name,x in shared.items():
    if x['unchanged'] or name.endswith('.npz') or name in ('historical_test_reference.py','resolved_configs.json','source_lineage.json'):continue
    p=ROOT/name;q=BASE/name
    patch.extend(difflib.unified_diff(q.read_text().splitlines(True) if q.exists() else [],p.read_text().splitlines(True),fromfile='a/experiments/typed_structure_early_cn_v1/'+name,tofile='b/experiments/typed_structure_early_cn_expansion_v1/'+name))
(ROOT/'minimal_changes.patch').write_text(''.join(patch))
old=read(BASE/'cost.json');measure={r['dataset']:r for r in old['measurements']}
devices=['GPU-dda1721a-d053-7371-3434-4c3c8502a1d4','GPU-994be766-a8bd-b90c-bf87-c667dfa8ab4d','GPU-e2284395-dd00-1e74-3eed-7f406fa69d5c','GPU-2db6cf6a-a8b5-58af-4e89-22bfc863a0fb','GPU-cddbf2e5-1af7-7219-ff58-38e6f4c44f69','GPU-360156ae-7a75-37f4-eaed-9a5500275409']
costs=[]
for task in PROTOCOL['tasks']:
    d=task['dataset'];h=measure[d];hist=h['history']
    if task['model']=='D':
        r=read(OUT/(d+'_D/summary.json'));step=max(x['normal2_seconds'] for x in r['updates']);train=3000*step+h['validation_save_setup_allowance_from_history_seconds'];chains=max(max(x['seconds']+x['posterior_seconds'] for x in r['inference'])*2000,float(hist['chain_seconds']))*1.05
        origin='new D engineering3 calls, t1/500/1000 full query no sampling; old validation/save/chain floor';peak=r['peak_allocated_mib'];reserved=r['peak_reserved_mib']
    elif task['fusion']=='early':
        step=h['normal2_max_measured_seconds'];train=h['train_extrapolation_seconds']+h['validation_save_setup_allowance_from_history_seconds'];chains=h['two_chains_extrapolation_seconds'];origin='reviewed E-early engineering reused; no repeat';peak=h['peak_allocated_mib'];reserved=h['peak_reserved_mib']
    else:
        step=float(hist['normal2_seconds'])/3000;train=float(hist['train_worker_seconds']);chains=float(hist['chain_seconds']);origin='old complete3000 E-late observed costs, new seed extrapolation';peak=float(hist['peak_allocated_mib']);reserved=float(hist['peak_reserved_mib'])
    costs.append(dict(**task,gpu_uuid=devices[task['slot']],normal2_component_seconds=step,train_including_validation_save_setup_seconds=train,two_serial_chains_seconds=chains,scientific_seconds=train+chains,source=origin,peak_allocated_mib=peak,peak_reserved_mib=reserved))
body=max(r['scientific_seconds'] for r in costs);plan=math.ceil(body*1.2+600+1200)
files,binding=source_binding();save(ROOT/'source_manifest.json',dict(source_sha256=binding,files=files))
cost=dict(status='passed',formal_authorized=False,source_sha256=binding,requested_budget_seconds=280*60,planned_seconds=plan,body_seconds=body,task_time_sum_seconds=sum(r['scientific_seconds'] for r in costs),concurrency_planning_factor=1.2,contingency_seconds=600,delivery_evaluation_seconds=1200,proposed_gpu_uuids=devices,schedule=jobs(),measurements=costs,limits=['not measured eight-way throughput','new six plus protected two may overlap; all one CPU thread, no DDP','no new full chain timed; state-dependent timing extrapolated','formal budget not active; original two deadlines unchanged','20 percent concurrency factor is planning only'],estimated_new_output_gib=8,launch_disk_minimum_gib=12)
assert plan<=280*60;save(ROOT/'cost.json',cost);save(OUT/'cost.json',cost);save(ev/'dry_run_queue.json',jobs())
protected=read(OUT/'protected_snapshot.json');check={p:sha(pathlib.Path(p))==h for p,h in protected.items()};assert all(check.values())
save(ev/'protected_unchanged.json',dict(checked_epoch=time.time(),files=check,all_unchanged=True))
save(ev/'failure_record.json',dict(scientific_failures=[],engineering_gpu_failures=[],preparation_incidents=['initial sparse-checkout setup failed; independent full worktree checkout used, original checkout unchanged','patch-format context/no-final-newline rejected before application; corrected patch, no computation affected','one read command used wrong current-directory-relative log path; corrected to absolute path'],temporary_optimizer_steps=6,new_formal_updates=0,new_complete_chains=0))
print(json.dumps(dict(source_sha256=binding,body_minutes=body/60,planned_minutes=plan/60,requested_minutes=280,changed_package=changed_package,artifacts=len(artifacts))))
