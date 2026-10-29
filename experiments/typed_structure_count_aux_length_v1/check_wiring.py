"""CPU metadata/hash/input-continuation checks, optimizer.step=0."""
import argparse,pathlib,json,time,copy,traceback,ast
import torch,numpy as np
from runtime_contract import ROOT,save,read,sha,jobs,PROTOCOL,verify_assets
from resume_io import parent_files,validate_parent
from prepare import load_case
from normal2 import pair_inputs
from probe import equal,input_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);start=time.time();rec=dict(status='running',optimizer_step_calls=0,model_forwards=0,generated_graphs=0)
    try:
        q=jobs()
        assert len(q['training'])==8 and len(q['sampling'])==24
        assert len({j['id'] for j in q['sampling']})==24
        assert sum(j['updates']-500 for j in q['training'])==32000
        assert sum(j['checkpoint']==3000 for j in q['sampling'])==16
        assert sum(j['checkpoint']==6000 for j in q['sampling'])==8
        assert all(j['dataset']=='PubMed' for j in q['sampling'] if j['checkpoint']==6000)
        verify_assets()
        rec['parents']=[];rec['next_input501_hashes']={};randoms={}
        for job in q['training']:
            row=parent_files(job);payload=torch.load(row['path'],map_location='cpu')
            expected=read(row['binding_path']);case,cfg=load_case(job['dataset']);cfg=dict(cfg,seed=job['train_seed'])
            validate_parent(payload,expected,job,cfg)
            rng=np.random.default_rng(0);rng.bit_generator.state=copy.deepcopy(payload['rng']['numpy'])
            inputs=pair_inputs(case,cfg,501,rng);ih=input_hash(inputs)
            rec['next_input501_hashes'][job['id']]=ih
            rec['parents'].append(dict(task=job['id'],checkpoint_sha256=row['sha256'],binding_sha256=row['binding_sha256'],
                updates=payload['updates'],ema_updates=payload['ema']['updates'],optimizer_states=len(payload['optimizer']['state']),
                input501_sha256=ih,input501_times=[x['t'] for x in inputs]))
            randoms[job['id']]=payload['rng'];del payload
        for ds,seeds in PROTOCOL['train_seeds'].items():
            for seed in seeds:
                a1=f'{ds}_s{seed}_CE';a2=a1+'_COUNT'
                assert equal(randoms[a1],randoms[a2])
                assert rec['next_input501_hashes'][a1]==rec['next_input501_hashes'][a2]
        parent=ROOT.parent/'typed_structure_count_aux_stage500_v1'
        files=[p for p in (ROOT/'package').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.npz')]
        files += [ROOT/n for n in ['ema.py','integration.py','stage.py','formula.json']]
        rec['unchanged_science_files']={str(p.relative_to(ROOT)):sha(p) for p in files}
        assert all(sha(parent/p.relative_to(ROOT))==sha(p) for p in files)
        old=ast.parse((ROOT.parent/'typed_structure_count_aux_v1/run_worker.py').read_text())
        new=ast.parse((ROOT/'run_worker.py').read_text())
        get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
        assert ast.dump(get(old))==ast.dump(get(new))
        rec['sampler_ast_identical']=True;rec['parent_rng_pairing_exact']=True
        rec['status']='passed'
    except BaseException as exc:rec.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    rec['seconds']=time.time()-start;save(pathlib.Path(a.output),rec);print(json.dumps({k:rec[k] for k in ['status','optimizer_step_calls','model_forwards','generated_graphs','seconds']}))
    return 0 if rec['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
