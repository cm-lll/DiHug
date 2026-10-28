"""Independent CPU cross-check of exported tables against copied records/states."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,json,pathlib,collections,hashlib
import numpy as np
def main():
    a=argparse.ArgumentParser();a.add_argument('--bundle',required=True);a=a.parse_args();p=pathlib.Path(a.bundle)
    def rows(n):return list(csv.DictReader((p/n).open()))
    vr=rows('validation_total.csv');vf=rows('validation_relations.csv');tr=rows('trajectory_total.csv');tf=rows('trajectory_relations.csv')
    assert len(vr)==216 and len(vf)==2160 and len(tr)==528 and len(tf)==5280
    for r in vr:
        raw=json.loads((p/f"evidence/{r['cell']}/validation.json").read_text());v=next(x for x in raw if x['updates']==int(r['updates']));v=next(x for x in v['rows'] if x['t']==int(r['noise_t']))
        assert float(r['total_CE'])==v['ce_sum_over_S'] and r['input_sha256']==v['input_sha256']
        assert abs(float(r['p0_sum'])-sum(x['expected_edges'] for x in v['family']))<1e-9
        for f in v['family']:
            c=next(x for x in vf if x['cell']==r['cell'] and x['updates']==r['updates'] and x['noise_t']==r['noise_t'] and int(x['family'])==f['family'])
            assert float(c['p0_sum'])==f['expected_edges'] and int(c['positive_count'])==f['positive_count'] and int(c['negative_count'])==f['negative_count']
            for key,rawkey in [('positive_CE_mean','positive_ce'),('negative_CE_mean','negative_ce')]:assert (None if c[key]=='' else float(c[key]))==f[rawkey]
    cache={};checked=0
    for r in tr:
        ds=r['dataset'];cell=r['cell'];seed=r['sample_seed'];key=(cell,seed)
        if key not in cache:cache[key]=np.load(p/f'evidence/{cell}/sample_{seed}/saved_states.npz',allow_pickle=False)
        z=cache[key];s=z['state'][list(z['state_t']).index(int(r['state_t']))];assert np.sum(s)==int(r['actual_edges'])
        assert hashlib.sha256(s.tobytes()).hexdigest()==r['state_array_sha256']
        ref=np.load(p/f'references/{ds}.npz',allow_pickle=False);types=ref['types'];sc=ref['schema'];ns=collections.Counter(types.tolist());offset=0;cs=[]
        for a,b in sc:
            n=ns[a]*(ns[a]-1)//2 if a==b else ns[a]*ns[b];cs.append(int(np.sum(s[offset:offset+n])));offset+=n
        assert offset==18336 and sum(cs)==int(r['actual_edges'])
        for c in tf:
            if c['cell']==cell and c['sample_seed']==seed and c['state_t']==r['state_t']:assert int(c['actual_edges'])==cs[int(c['family'])]
        checked+=1
    # All two-sample means must have exactly two participants; no success-only means.
    for r in rows('trajectory_two_sample_means.csv'):
        matched=[x for x in tr if x['dataset']==r['dataset'] and x['train_seed']==r['train_seed'] and x['arm']==r['arm'] and x['state_t']==r['state_t']];assert len(matched)==2
        assert float(r['actual_edges_mean'])==sum(int(x['actual_edges']) for x in matched)/2
    result=dict(status='passed',validation_records=216,relation_validation_records=2160,saved_states=checked,model_loads=0,torch_imports=0,model_forwards=0,new_graphs=0)
    print(json.dumps(result));(p/'crosscheck.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
