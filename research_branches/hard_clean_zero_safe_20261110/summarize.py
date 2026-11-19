"""Read-only CPU audit and complete paired cost/graph-statistic tables."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/hard_clean_zero_safe_20261110'
sys.path.insert(0,str(ROOT/'research_branches/hard_clean_pilot_v2_20261110'))
from runtime import sha,g
torch.set_num_threads(1)


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());receipt=json.loads((OUT/'receipt.json').read_text())
    tables=[];audit=[];aggregate=[]
    for row in protocol['tasks']:
        folder=OUT/row['task'];r=json.loads((folder/'result.json').read_text())
        old=torch.load(row['stop_file'],map_location='cpu')
        new=torch.load(folder/'temporary_complete.pt',map_location='cpu')
        assert new['state']['updates']==old['state']['updates']+1==new['state']['ema']['updates']
        fixed=torch.load(row['evaluation_file'],map_location='cpu')['inputs']
        replay=np.load(folder/'failed_batch_replay.npz')
        ar=np.random.default_rng();ar.bit_generator.state=old['auxiliary_rng']
        bits=(ar.random(replay['bits'].shape)<replay['probability']).astype(np.uint8)
        assert np.array_equal(bits,replay['bits'])
        assert ar.bit_generator.state==new['auxiliary_rng'],'auxiliary RNG not paired'
        assert g.digest(new['state']['rng'])==r['main_rng_sha256']
        before=json.loads((folder/'before.json').read_text());after=json.loads((folder/'after.json').read_text())
        for j,(left,right) in enumerate(zip(before,after)):
            assert left['input_sha256']==right['input_sha256'] and left['t']==right['t']
            d=dict(task=row['task'],dataset=row['dataset'],seed=row['seed'],update=r['update'],t=left['t'])
            q=fixed[j]['query'][:,np.argsort(fixed[j]['ids'])]
            # Both datasets have a fixed 192-node inventory, including isolates.
            n=192
            for label,er in [('before',left),('after',right)]:
                pred=np.load(folder/f'{label}_t{left["t"]}.npz');stats=[]
                assert np.array_equal(pred['ids'],replay['ids'])
                uniforms=np.random.default_rng(row['seed']+810001+100+j).random(pred['bits'].shape)
                assert np.array_equal(uniforms<pred['probability'],pred['bits'])
                for bb in pred['bits']:
                    i,k=q[:,bb.astype(bool)]
                    adj=np.zeros((n,n),dtype=np.int64);adj[i,k]=1;adj[k,i]=1
                    nc,labels=connected_components(csr_matrix(adj),directed=False)
                    tri=int(np.sum((adj@adj)*adj)//6)
                    stats.append([bb.sum(),nc,np.bincount(labels).max()/n,tri,
                                  (adj.sum(1)==0).sum(),adj.sum(1).max()])
                for h,name in enumerate(['edges','components','LCC_fraction','triangles','isolates','hub']):
                    d[f'{label}_{name}']=float(np.mean(stats,axis=0)[h])
                for h,name in enumerate(['relation_count','joint_degree','present_triangles','absent_triangles','components']):
                    d[f'{label}_cost_{name}']=er['hard_cost_mean'][h]
                for name in ('CE','positive_CE','negative_CE'):d[f'{label}_{name}']=er[name]
            tables.append(d)
        subset=tables[-len(before):]
        a=dict(task=row['task'],update=r['update'],old_MGDA_norm=r['old_MGDA_norm'],new_gradient_norm=r['new_gradient_norm'],
            AdamW_norm=r['actual_AdamW_norm'],mean_cost_before=r['mean_cost_before'],mean_cost_after=r['mean_cost_after'])
        for label in ('before','after'):
            for name in ('edges','components','LCC_fraction','triangles','isolates','hub','positive_CE','negative_CE'):
                a[f'{label}_{name}']=float(np.mean([v[f'{label}_{name}'] for v in subset]))
        aggregate.append(a)
        audit.append(dict(task=row['task'],main_rng_saved_exact=True,auxiliary_draws_and_saved_rng_exact=True,
            same_evaluation_uniforms=True,old_failed_cost_max_difference=r['old_cost_max_difference'],
            counter_and_ema_correct=True,state_reload_bitwise=r['save_reload_boundary_exact']))
    for path,h in protocol['bindings'].items():assert sha(path)==h
    with (OUT/'paired_by_time.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    result=dict(status='combination_stall_repaired_not_safe_for_long_training',conditions=aggregate,audit=audit,
        old_bound_files_unchanged=len(protocol['bindings']),cost_names=['relation_count','joint_degree','present_triangles','absent_triangles','components'],
        actual_calls={k:receipt[k] for k in ('forwards','backwards','optimizer_updates')},
        GPU_compute_process_seconds=receipt['elapsed_seconds'],GPU_hours_conservative=(receipt['elapsed_seconds']+60)/3600,
        accounting='60 additional seconds conservatively covers import/exit overhead excluded from worker timer; GPU idle after exit',
        peak_reserved_MiB=receipt['peak_reserved_MiB'],full_generation_chains=0,automatic_long_training=False)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
