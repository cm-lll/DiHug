"""CPU-only assets for two new paired seeds. Does not launch GPU computation."""
import os,json,hashlib,shutil,time,copy,tarfile,subprocess,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent
PARENT=ROOT/'results/structure_count_unit_formal_20261107T163906'
OUT=ROOT/'results/structure_count_unit_extension_20261107T173846'
os.environ.update(STRUCTURE_ASSETS=str(ROOT),STRUCTURE_RUN=str(OUT),CUDA_VISIBLE_DEVICES='')
from runtime import science as g
from experiment_contract import initialization,reset_training_streams
from probe import input_hash
from worker import atomic_checkpoint
from runtime.objective import make_objective
def read(p):return json.loads(p.read_text())
def save(p,z):p.write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    g.torch.set_num_threads(1);g.torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False)
    for n in ['prepared','logs','leases']:(OUT/n).mkdir()
    save(OUT/'approval.json',dict(authorized=True,user_instruction='好的',scope='four additional models, two new dataset seeds, PB/UNIT matched,3000updates and16chains',launcher_pid=None,combined_GPU_hours=13.5,additional_GPU_hours=6.5,deadline_confirmed=False))
    # The extension's new absolute deadline is pending; no launcher is authorized
    # by this provisional ledger. Parent ledger and source are read-only.
    save(OUT/'ledger.json',dict(start_epoch=1794044326,science_stop_epoch=1794048546,deadline_epoch=1794050346,GPU_hours_limit=6.5,combined_GPU_hours_limit=13.5,parent_run=str(PARENT),deadline_confirmed=False))
    save(OUT/'preregistration.json',dict(training_seeds={'IMDB':2026110811,'PubMed':2026110821},arms=['PB','UNIT'],sample_seeds=[2026102821,2026102822,2026110231,2026110232],updates=3000,selected_before_new_results=True,metrics='same frozen full original and LCC evaluator; per training model four sample mean and sample SD; new seed and previous panel reported separately',no_formula_change=True))
    sources=read(PARENT/'source_binding.json')
    for p,h in sources.items():assert sha(Path(p))==h,p
    old_tasks=read(PARENT/'tasks.json');tasks=[];checks=[]
    for ds,seed in [('IMDB',2026110811),('PubMed',2026110821)]:
        template=next(x for x in old_tasks if x['dataset']==ds);cfg=copy.deepcopy(template['config']);cfg['seed']=seed
        m,case,base,proof=initialization(ds,'A',seed,None)
        assert cfg['model']==base['model'] and cfg['optimizer']==base['optimizer']
        opt=g.torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        ema=g.ParameterEMA(m);rng,streams=reset_training_streams(seed,ds,'cpu')
        boundary=g.TrainingBoundary(m,opt,ema,rng,'cpu',0)
        binding=dict(dataset=ds,train_seed=seed,arm='A',config=cfg,updates=0,initialization=proof,streams=streams)
        parent=OUT/'prepared'/f'{ds}_{seed}_initial_cpu.pt';payload=boundary.checkpoint(binding);atomic_checkpoint(parent,payload)
        restored=g.torch.load(parent,map_location='cpu')
        assert g.digest(payload)==g.digest(restored)
        assert not payload['optimizer']['state'] and payload['updates']==payload['ema']['updates']==0
        # CPU arithmetic-only adapter equivalence; no denoiser call.
        objective=make_objective(case,cfg,ds,'PB');owner=objective.pb
        ids=g.np.flatnonzero(g.np.repeat(case['active'],case['space'].counts))
        inp=g.one_input(case,cfg,1,rng)
        p=g.torch.full((len(ids),),.25,dtype=g.torch.float64)
        a,da=objective.pb(p,inp['query'],inp['f'])
        b,db=objective(p,inp['query'],inp['f'],inp,logits=None)
        assert g.torch.equal(a,b) and g.torch.equal(da,db)
        objective.assert_binding(objective.objective_binding())
        g.rng_restore(payload['rng'],rng,'cpu');contract=[]
        for u in range(1,3001):
            inp=g.one_input(case,cfg,u,rng)
            contract.append(dict(update=u,t=inp['t'],slots=len(inp['ids']),sha256=input_hash([inp])))
        save(OUT/'prepared'/f'input_contract_{ds}_{seed}.json',contract)
        for arm in ['PB','UNIT']:
            tasks.append(dict(task_id=f'{ds}_{seed}_{arm}',dataset=ds,train_seed=seed,arm=arm,gpu=4+len(tasks),config=cfg,initial=str(parent),initial_sha256=sha(parent),CUDA_stream_seed=streams['seeds']['torch_cuda']))
        checks.append(dict(dataset=ds,seed=seed,initial_sha256=sha(parent),model_digest=g.digest(payload['model']),CPU_RNG_digest=g.digest(payload['rng']),paired_same_parent=True,empty_optimizer=True,EMA_updates=0,updates=0,PB_adapter_exact=True,input_contract_sha256=sha(OUT/'prepared'/f'input_contract_{ds}_{seed}.json'),input_updates=3000,CUDA_state='initialized once per worker from fixed paired CUDA seed after model construction, then sealed at zero'))
        del m,opt,ema,boundary,payload,restored
    shutil.copy2(PARENT/'prepared/sampling_contracts.json',OUT/'prepared/sampling_contracts.json')
    save(OUT/'tasks.json',tasks)
    assert not g.torch.cuda.is_initialized()
    save(OUT/'prepared/receipt.json',dict(passed=True,checks=checks,CUDA_initialized=False,denoiser_calls=0,reused_engineering_receipt=str(PARENT/'engineering/receipt.json'),reused_engineering_sha256=sha(PARENT/'engineering/receipt.json'),ordinary_recompute_tolerance_passed=False))
    for p in SRC.rglob('*.py'):sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources)
    save(OUT/'source_version.json',dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_manifest_sha256=sha(OUT/'source_binding.json'),identity='isolated uncommitted source SHA manifest; parent commit is provenance only'))
    before=ROOT/'research_branches/structure_count_unit_formal_review_20261107';diff=[]
    for p in SRC.rglob('*.py'):
        rel=p.relative_to(SRC);old=before/rel
        diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile=str(old),tofile=str(p)))
    (OUT/'runtime_diff.patch').write_text(''.join(diff))
    with tarfile.open(OUT/'frozen_source.tar.gz','w:gz') as tar:
        for p in sources:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    print(json.dumps(dict(prepared=True,out=str(OUT),models=len(tasks),GPU_calls=0,deadline_confirmation_required=True)))
if __name__=='__main__':main()
