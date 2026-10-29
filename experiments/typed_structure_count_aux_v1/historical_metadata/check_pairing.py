"""Initialization/input contract only; no forward, backward, step or chain."""
import argparse,pathlib,copy,hashlib,json,gc
from model_io import configure,fresh,tensor_hash
import torch,numpy as np
from runtime_contract import PROTOCOL,save,sha,source_binding
from probe import rng_get,equal,input_hash
from normal2 import pair_inputs
from ema import ParameterEMA
from integration import checkpoint_payload

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    env=configure();rows=[]
    for dataset,seed in PROTOCOL['new_train_seeds'].items():
        pair=[]
        for arm in ('E_lateCN','E_earlyCN'):
            m,opt,case,cfg,rng,proof=fresh(dataset,arm,seed);ema=ParameterEMA(m)
            initial=checkpoint_payload(m,opt,ema,0,rng_get(rng,'cuda'),dict(dataset=dataset,arm=arm,seed=seed,source_sha256=source_binding()[1],proof=proof,engineering_only=True))
            path=out/(dataset+'_'+arm+'_initial.pt');torch.save(initial,path)
            fingerprints=[input_hash(pair_inputs(case,cfg,position,rng)) for position in (1,2,3)]
            sample=[]
            for ss in PROTOCOL['sample_seeds']:
                gen=np.random.default_rng(ss);families=np.repeat(np.arange(case['n_families']),case['space'].counts)
                prior=(gen.random(case['space'].total)<case['prior'][families]).astype(np.uint8)
                # No network/reverse step; record paired initial noise and random draws only.
                draws=gen.random(case['space'].total)
                sample.append(dict(sample_seed=ss,prior_sha256=hashlib.sha256(prior.tobytes()).hexdigest(),next_draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest()))
            pair.append(dict(arm=arm,state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},rng=initial['rng'],proof=proof,training_input_hashes=fingerprints,sample_contract=sample,path=str(path),sha256=sha(path)))
            del m,opt,ema,initial;gc.collect();torch.cuda.empty_cache()
        late,early=pair;checks={}
        for k,v in early['state'].items():
            if k=='cn_edge_projection.weight':checks[k]=bool(torch.count_nonzero(v)==0)
            else:checks[k]=torch.equal(v,late['state'][k][:,:739] if k=='joint_head.0.weight' else late['state'][k])
        ok=all(checks.values()) and equal(late['rng'],early['rng']) and late['training_input_hashes']==early['training_input_hashes'] and late['sample_contract']==early['sample_contract'] and late['proof']['construction']['late_initial_content_sha256']==early['proof']['construction']['late_initial_content_sha256']
        row=dict(dataset=dataset,seed=seed,passed=ok,common_tensors_exact=checks,all_rng_exact=equal(late['rng'],early['rng']),training_inputs_equal=late['training_input_hashes']==early['training_input_hashes'],sampling_contract_equal=late['sample_contract']==early['sample_contract'],arms=[])
        for x in pair:row['arms'].append({k:v for k,v in x.items() if k not in ('state','rng')})
        rows.append(row);save(out/'receipt.json',dict(status='passed' if all(x['passed'] for x in rows) else 'failed',environment=env,rows=rows,optimizer_steps=0,forwards=0,complete_chains=0))
        if not ok:raise AssertionError('early/late paired initial/input mismatch')
    return 0
if __name__=='__main__':raise SystemExit(main())
