"""Read-only old-initial-state comparison; constructs no trained model/output.

Old initial checkpoints are caller-specified and checked against archived receipts.
No optimizer.step, GPU operation, dictionary fitting or graph generation.
"""
import argparse,pathlib,json,hashlib,gc
import numpy as np
import torch
from prepare import save,sha
from experiment_contract import initialization,read_dictionary,reset_training_streams
from normal2 import pair_inputs
from probe import input_hash

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--round1',required=True);ap.add_argument('--round2',required=True)
    ap.add_argument('--dictionaries',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);rows=[]
    for index,oldroot in enumerate([pathlib.Path(a.round1),pathlib.Path(a.round2)]):
        for dataset in ['IMDB','PubMed']:
            seed=([2026102811,2026102834] if dataset=='IMDB' else [2026102812,2026102833])[index]
            basis,receipt=read_dictionary(a.dictionaries,dataset)
            old={};bindings={};hashes={}
            for arm in 'ABC':
                folder=oldroot/(dataset+'_'+arm);h=json.loads((folder/'initial.sha256.json').read_text())['sha256']
                assert sha(folder/'initial.pt')==h
                ck=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
                assert ck['updates']==0 and ck['binding']['arm']==arm and ck['binding']['config']['seed']==seed
                assert ck['binding']['dictionary_sha256']==receipt['basis_sha256']
                old[arm]=ck['model'];bindings[arm]=ck['binding'];hashes[arm]=h
            for arm in 'DE':
                model,case,cfg,proof=initialization(dataset,arm,seed,basis if arm=='E' else None)
                state=model.state_dict();comparisons={}
                for oldarm in 'ABC':
                    matched=[];absent=[]
                    for name,value in old[oldarm].items():
                        if name not in state:absent.append(name);continue
                        target=state[name]
                        if name=='joint_head.0.weight':target=target[:,:value.shape[1]]
                        assert target.shape==value.shape and torch.equal(target,value),(dataset,seed,arm,oldarm,name)
                        matched.append(name)
                    comparisons[oldarm]=dict(exact_matched=matched,not_present_in_new_arm=absent,initial_sha256=hashes[oldarm])
                for key in ['model','optimizer','precision','T','rounds','schedule_offset','tf32','feature_spec','datasets']:
                    assert cfg[key]==bindings['A']['config'][key],key
                rng,rngproof=reset_training_streams(seed,dataset,'cpu')
                archived_rng=json.loads((oldroot/(dataset+'_A')/'training_rng_reset.json').read_text())
                assert rngproof['seeds']==archived_rng['seeds']
                for name in ['numpy','global_numpy','python','torch']:
                    assert rngproof['state_sha256'][name]==archived_rng['state_sha256'][name],name
                lines=(oldroot/(dataset+'_A')/'training.jsonl').read_text().splitlines()[:3]
                input_checks=[]
                for j,line in enumerate(lines,1):
                    inp=pair_inputs(case,dict(cfg,seed=seed),j,rng)
                    actual=input_hash(inp);assert actual==json.loads(line)['input_sha256']
                    input_checks.append(actual)
                label=f'{dataset}_{seed}_{arm}'
                save(out/(label+'.json'),dict(initialization=proof,comparisons=comparisons,rng=rngproof,
                    first_three_training_inputs=input_checks,basis_sha256=receipt['basis_sha256'],
                    CUDA_rng='not recreated in CPU audit; first-seed GPU engineering receipt compared separately',
                    seed=seed,dataset=dataset,arm=arm,parameters=proof['parameters'],
                    purpose='comparison-only untrained reconstruction; NOT a resumed trained model'))
                rows.append(dict(label=label,passed=True,parameters=proof['parameters'],old_initial_hashes=hashes))
                del model,state;gc.collect()
            del old,ck;gc.collect()
    save(out/'summary.json',dict(passed=True,cells=rows,old_trained_weights_used=False,optimizer_steps=0,
        interpretation='paired common and corresponding projection initial tensors; architecture/parameter count differs intentionally'))

if __name__=='__main__':main()
