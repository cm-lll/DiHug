"""New shared A initial banks and complete input plans, CPU only/no forward."""
from .formal_runtime import *
from experiment_contract import initialization,reset_training_streams
import shutil

def main():
    setup();tasks=[];proofs=[]
    old=read(ASSETS/'results/pb_degree_formal_20261106T170600/tasks.json')
    for ds,seed in [('IMDB',2026110711),('PubMed',2026110721)]:
        model,case,base,proof=initialization(ds,'A',seed,None)
        cfg=copy.deepcopy(next(x['config'] for x in old if x['dataset']==ds));cfg['seed']=seed
        assert cfg['model']==base['model'] and cfg['optimizer']==base['optimizer']
        bank=OUT/'prepared'/f'{ds}_{seed}_bank.pt'
        atomic_checkpoint(bank,dict(model=g.common.cpu_tree(model.state_dict()),config=cfg,dataset=ds,seed=seed,proof=proof))
        rng,rr=reset_training_streams(seed,ds,'cpu');rows=[]
        for u in range(1,3001):
            check();inp=g.one_input(case,cfg,u,rng)
            rows.append(dict(update=u,t=inp['t'],slots=len(inp['ids']),sha256=input_hash([inp])))
        save(OUT/'prepared'/f'input_contract_{ds}_{seed}.json',rows)
        for arm in ['A','PB']:
            tasks.append(dict(task_id=f'{ds}_{seed}_{arm}_late',dataset=ds,train_seed=seed,arm=arm,gpu=len(tasks),config=cfg,bank=str(bank),bank_sha256=sha(bank),initial=str(OUT/'prepared'/f'{ds}_{seed}_initial.pt'),initial_sha256=None))
        proofs.append(dict(dataset=ds,seed=seed,initial_proof=proof,input_count=len(rows),CPU_rng=rr))
        del model
    save(OUT/'tasks.json',tasks);save(OUT/'prepared/CPU_receipt.json',dict(passed=True,proofs=proofs,new_forward_calls=0))
    # Sampling protocol unchanged by model/seed; keep the complete sealed plan.
    sampling=ASSETS/'results/pb_degree_formal_20261106T170600/prepared/sampling_contracts.json'
    shutil.copy2(sampling,OUT/'prepared/sampling_contracts.json')
    sources=read(ASSETS/'results/pb_degree_formal_20261106T170600/source_binding.json')
    for p,h in sources.items():assert sha(p)==h,p
    # Include both reused dependencies and every isolated new Python file.
    for p in ROOT.rglob('*.py'):sources[str(p)]=sha(p)
    save(OUT/'source_binding.json',sources)
    print('CPU preparation complete: shared banks,6000 inputs,1000-step sampling plans')

def seal_cuda():
    from experiment_contract import reset_training_streams
    tasks=read(OUT/'tasks.json')
    for ds in ['IMDB','PubMed']:
        row=next(x for x in tasks if x['dataset']==ds);cfg=row['config'];seed=row['train_seed']
        assert sha(row['bank'])==row['bank_sha256']
        bank=torch.load(row['bank'],map_location='cpu')
        assert bank['dataset']==ds and bank['seed']==seed and bank['config']==cfg
        data=ASSETS/SCI.relative_to(SOURCE)/'package'/cfg['datasets'][ds]['file']
        assert sha(data)==cfg['datasets'][ds]['sha256']
        with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        m=g.StructuralDenoiser(case,cfg,'A').cuda();m.load_state_dict(bank['model'],strict=True)
        assert g.digest(m.state_dict())==g.digest(bank['model'])
        o=torch.optim.AdamW(m.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
        e=g.ParameterEMA(m);r,_=reset_training_streams(seed,ds,'cuda')
        b=g.TrainingBoundary(m,o,e,r,'cuda',0)
        binding=dict(dataset=ds,train_seed=seed,arm='A',config=cfg,bank_sha256=row['bank_sha256'])
        payload=b.checkpoint(binding);assert payload['updates']==0 and payload['ema']['updates']==0 and not payload['optimizer']['state']
        atomic_checkpoint(Path(row['initial']),payload)
        for task in tasks:
            if task['dataset']==ds:task['initial_sha256']=sha(row['initial'])
        del m,o,e,r,b,payload;torch.cuda.empty_cache()
    save(OUT/'tasks.json',tasks)

if __name__=='__main__':main()
