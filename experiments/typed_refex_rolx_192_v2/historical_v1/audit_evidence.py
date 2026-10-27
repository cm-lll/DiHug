"""Offline audit: arrays, sources, update caps, original posterior. No network."""
import argparse,json,pathlib,hashlib
import numpy as np
from prepare import ROOT,load_case,sha,save
from hsgd.bw_preflight.common import train_input,schedule,plan_for
from hsgd.kernel import MarginalKernel

def audit(results):
    results=pathlib.Path(results);rows=[];counts={}
    for run in ['gpu_probe','gpu_probe_corrected','cpu_remaining','gpu_forward_only']:
        folder=results/run
        if not folder.exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        for key,row in summary['cases'].items():
            actual=(folder/key/'updates.jsonl')
            updates=[json.loads(s) for s in actual.read_text().splitlines()] if actual.exists() else []
            counts[key]=counts.get(key,0)+len(updates)
            assert all(u['edge_head_max_update']>0 and np.isfinite(u['gradient_norm']) for u in updates)
            for artifact in list((folder/key).glob('*query*.npz')):
                dataset,arm=key.split('_');c,cfg=load_case(dataset)
                with np.load(artifact,allow_pickle=False) as z:state,p,q,cover=[z[k] for k in ['state','p0','q','coverage']]
                # Deterministically reproduce original forward-noise input, no network or graph generation.
                inp=train_input(c,1000,20,cfg['seed'],50,np.random.default_rng(2026102712),force_t=500)
                expected_state=np.zeros(c['space'].total,dtype=np.uint8);f,loc=c['space'].encode(inp['edges']);expected_state[c['space'].offsets[f]+loc]=1
                assert np.array_equal(state,expected_state)
                allf=np.repeat(np.arange(c['n_families']),c['space'].counts);active=c['active'][allf]
                assert np.array_equal(cover,active.astype(np.uint8))
                assert np.isnan(p[~active]).all() and np.isnan(q[~active]).all()
                assert np.isfinite(p[active]).all() and ((p[active]>=0)&(p[active]<=1)).all()
                recomputed=MarginalKernel(c['prior'],schedule(1000)).reverse(p[active],state[active],allf[active],499,500)
                np.testing.assert_array_equal(q[active],recomputed)
                rows.append({'path':str(artifact),'sha256':sha(artifact),'all_active_once':True,'posterior_exact':True,'input_replayed':True})
    assert all(n<=5 for n in counts.values())
    assert counts==json.loads((results/'executed_update_counts.json').read_text())
    binding=json.loads((ROOT/'source_binding.json').read_text())
    for name,row in binding['files'].items():
        local=ROOT/pathlib.Path(name).relative_to('historical_source')
        assert sha(local)==row['sha256']
    for name,h in binding['audit_reference_only'].items():assert sha(ROOT/'audit_reference'/name)==h
    return dict(status='passed_offline_audit',arrays=rows,executed_engineering_updates=counts,
                formal_updates=0,generated_graphs=0,source_original_hsgd_unchanged=True,
                note='Does not override CPU/GPU or continuation failures; NNLS/NMF quality not certified')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=audit(a.results);save(a.output,value);print(json.dumps(value,indent=2))
