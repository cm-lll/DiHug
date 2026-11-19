"""CPU-only immutable asset, RNG, query-support and budget binding."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,datetime,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as s
np=s.np
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    out=ROOT/'results/block_generation_panel_20261112T015400';out.mkdir(exist_ok=False);(out/'logs').mkdir();(out/'chains').mkdir()
    audit=ROOT/'results/block_context_audit_20261112T014400';old=read(audit/'protocol.json')
    assert read(audit/'receipt.json')['status']=='completed'
    for path,h in old['bindings'].items():assert sha(path)==h,path
    models={};contract={};bindings=dict(old['bindings']);jobs=[];reused=[];checks=[]
    seeds=[2026111341,2026111342,2026111343,2026111344];rounds=[1,10,20]
    for ds in ['IMDB','PubMed']:
        row=read(audit/ds/'binding.json');models[ds]=row
        parent=s.torch.load(row['parent_readout'],map_location='cpu',weights_only=False)
        z=s.torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert s.state_digest(z['model'])==row['model_digest'] and z['updates']==2
        assert z['binding']['dataset']==ds and z['binding']['train_seed']==row['train_seed']
        assert z['ema']['updates']==2 and z['recovery']['exact_resume_allowed']
        del parent,z
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=s.w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        fam=np.repeat(np.arange(case['n_families']),case['space'].counts)
        contract[ds]={}
        for ss in seeds:
            rng=np.random.default_rng(ss);initial_u=rng.random(case['space'].total);initial=(initial_u<case['prior'][fam]).astype(np.uint8)
            entry=dict(initial_hash=s.digest(initial),initial_uniform_hash=s.digest(initial_u),step_uniform_hashes=[s.digest(rng.random(case['space'].total)) for _ in range(1000)])
            contract[ds][str(ss)]=entry
            for r in rounds:
                jobid=f'{ds}_R{r}_S{ss}';reuse=r==1 and ss in seeds[:2]
                directory=(ROOT/f'results/three_arm_learning_pilot_20261111T211641/{ds}_T_eager/final_sample_{ss}') if reuse else out/'chains'/jobid
                job=dict(id=jobid,dataset=ds,rounds=r,sample_seed=ss,train_seed=row['train_seed'],reused=reuse,directory=str(directory))
                jobs.append(job)
                inventory=s.w.common.query_inventory(case,s.w.common.plan_for(case,[ss,500,61],r))
                checks.append(dict(job=jobid,query_contract=inventory))
                if reuse:
                    ident=read(directory/'identity.json');sm=read(directory/'summary.json')
                    steps=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
                    assert ident['model_hash']==row['model_digest'] and ident['updates']==2 and ident['R']==1 and ident['T']==1000
                    assert ident['initial_hash']==entry['initial_hash'] and ident['sample_seed']==ss and ident['train_seed']==row['train_seed']
                    assert [x['time'] for x in steps]==list(range(1000,0,-1))
                    assert [x['uniform_hash'] for x in steps]==entry['step_uniform_hashes']
                    assert all(x['exact_coverage'] and x['covered_slots']==inventory['active_slots'] for x in steps)
                    assert sm['status']=='complete' and sha(directory/'graph.npz')==sm['graph_sha256']
                    for name in ['identity.json','summary.json','steps.jsonl','graph.npz']:bindings[str(directory/name)]=sha(directory/name)
                    reused.append(job)
    costs=read(audit/'cost_extrapolation.json')['estimates']
    new=[j for j in jobs if not j['reused']]
    for j in new:
        minutes=(8.5 if j['dataset']=='IMDB' else 11.5) if j['rounds']==1 else costs[j['dataset']+'_R'+str(j['rounds'])]['chain_minutes_linear_mean']
        j['nominal_minutes']=minutes+0.5
    new.sort(key=lambda j:(-j['nominal_minutes'],j['id']))
    loads=[0.]*6
    for j in new:loads[loads.index(min(loads))]+=j['nominal_minutes']
    start=datetime.datetime.fromisoformat('2026-11-12T01:54:00+08:00').timestamp()
    gpu_hours=sum(j['nominal_minutes'] for j in new)/60;elapsed=(time.time()-start)/60
    budget=dict(nominal_GPU_hours=gpu_hours,nominal_queue_minutes=max(loads),planned_per_device_minutes=loads,
        margin_factor=1.25,with_margin_GPU_hours=gpu_hours*1.25,with_margin_completion_minutes_from_start=elapsed+max(loads)*1.25,
        actual_chain_unmeasured=True,free_running_density_and_concurrency_unmeasured=True)
    assert gpu_hours*1.25<14 and elapsed+max(loads)*1.25<160,'full panel cannot fit: no automatic scope reduction'
    for name,mod in list(sys.modules.items()):
        path=getattr(mod,'__file__',None)
        if path and path.endswith('.py') and str(ROOT) in str(Path(path).resolve()):bindings[str(Path(path).resolve())]=sha(path)
    # Bind evaluation sources too; they never write to historical result directories.
    extra=[ROOT/'research_branches/terminal_structure_real_20261109/evaluate.py',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/heat_solver_candidate.py']
    extra+=list((ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval').rglob('*.py'))
    for path in extra:bindings[str(path)]=sha(path)
    for path in Path(__file__).parent.glob('*.py'):
        py_compile.compile(str(path),doraise=True);bindings[str(path)]=sha(path)
    devices=[dict(index=i,uuid=u) for i,u in [(2,'GPU-dda1721a-d053-7371-3434-4c3c8502a1d4'),(3,'GPU-994be766-a8bd-b90c-bf87-c667dfa8ab4d'),(4,'GPU-e2284395-dd00-1e74-3eed-7f406fa69d5c'),(5,'GPU-2db6cf6a-a8b5-58af-4e89-22bfc863a0fb'),(6,'GPU-cddbf2e5-1af7-7219-ff58-38e6f4c44f69'),(7,'GPU-360156ae-7a75-37f4-eaed-9a5500275409')]]
    save(out/'random_contract.json',contract);bindings[str(out/'random_contract.json')]=sha(out/'random_contract.json')
    save(out/'protocol.json',dict(start_epoch=start,science_stop_epoch=start+160*60,delivery_epoch=start+180*60,max_GPU_hours=14,
        devices=devices,models=models,bindings=bindings,all_jobs=jobs,new_jobs=new,reused=reused,rounds=rounds,sample_seeds=seeds,
        snapshot_times=[1000,950,750,500,250,50,1],budget=budget,training_updates=0,readout='fixed raw terminal update2',
        gate='372 fixed-state forward calls passed; full coverage/frozen state/RNG; total nominal+25% fits whole panel',
        primary_metrics=['degree_W1','joint_TV','triangle_abs_error','MP3_abs_error','MP3_composition_TV','LCC'],
        retain_all_metrics=True,limitations=['one training model per dataset','four sample seeds are not independent training replicates','original full-chain learning still in progress on GPUs0/1; do not change it','no claim of query-block backward equivalence or large-graph feasibility']))
    save(out/'CPU_preflight.json',dict(status='passed',query_support_checks=len(checks),checks=checks,reused_graphs=4,matched_1000_uniform_records_per_old_graph=True,model_calls=0,GPU=0,budget=budget,bindings=len(bindings)))
    print(out);print(json.dumps(budget,indent=2))
if __name__=='__main__':main()
