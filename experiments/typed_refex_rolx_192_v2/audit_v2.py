"""Read-only evidence audit; no network forward or optimization."""
import argparse,json,pathlib,sys,hashlib,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'package'))
from prepare import load_case,ROOT,sha
from hsgd.bw_preflight.common import schedule
from hsgd.kernel import MarginalKernel

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);args=p.parse_args();out=pathlib.Path(args.evidence)
    s=json.loads((out/'probe/summary.json').read_text());counts=json.loads((out/'executed_update_counts.json').read_text())
    result=dict(status='passed',checks={},scientific_training=False,generated_graphs=0)
    for key,row in s['cases'].items():
        if row['status']!='completed_engineering_only':result['checks'][key]={'status':'incomplete'};result['status']='incomplete';continue
        dataset,arm=key.split('_');case,cfg=load_case(dataset);folder=out/'probe'/key
        with np.load(folder/'full_query.npz') as z:
            expected=np.flatnonzero(np.repeat(case['active'],case['space'].counts))
            assert np.array_equal(np.sort(z['ids']),expected) and len(np.unique(z['ids']))==len(expected)
            f,_=case['space'].global_to_local(z['ids']) if hasattr(case['space'],'global_to_local') else (None,None)
            q=MarginalKernel(case['prior'],schedule(1000)).reverse(z['p0'],z['state'][z['ids']],z['families'],499,500)
            np.testing.assert_array_equal(q,z['q']);assert np.isfinite(q).all()
        assert row['actual_updates']==counts[key]==3 and row['load_exact']
        if (folder/'after1.pt').exists():assert sha(folder/'after1.pt')==row['checkpoint_sha256']
        else:
            omitted=json.loads((out/'local_engineering_checkpoints.json').read_text())
            assert omitted[str((folder/'after1.pt').relative_to(out))]['sha256']==row['checkpoint_sha256']
        rr=json.loads((folder/'restore.json').read_text())
        expected_pass=rr['loss']['passes'] and all(v['passes'] for item in rr['inputs'] for v in item.values()) and all(v['passes'] for v in rr['gradients'].values()) and all(v['passes'] for v in rr['parameters'].values()) and rr['optimizer_exact'] and rr['rng_exact']
        assert rr['passes']==expected_pass
        for d in ['cpu_repeat','same_device','cross_backend']:
            report=json.loads((folder/(d+'.json')).read_text());assert report['gradients'] and len(report['inputs'])==2
        result['checks'][key]=dict(status='audited',coverage=len(expected),posterior_exact=True,updates=3,loaded_exact_receipt=True,checkpoint_bytes_rehashed=(folder/'after1.pt').exists(),restore_passed=rr['passes'])
    for dataset in ['IMDB','PubMed']:
        hashes={v['input_sha256'] for k,v in s['cases'].items() if k.startswith(dataset)};assert len(hashes)==1
    result['total_actual_updates']=sum(counts.values());assert result['total_actual_updates']<=18
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
