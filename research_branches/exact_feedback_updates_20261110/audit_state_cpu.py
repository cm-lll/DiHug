"""Independent CPU checks of serialized auxiliary streams and asset integrity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[]
    for path,h in json.loads((OUT/'bindings.json').read_text()).items():assert rt.sha(path)==h,path
    freeze=json.loads((OUT/'evaluation_freeze.json').read_text());assert rt.sha(ROOT/freeze['script'])==freeze['sha256']
    for task in protocol['tasks']:
        folder=OUT/f"{task['dataset']}_{task['seed']}"
        if not (folder/'checks.json').exists():continue
        updates=json.loads((folder/'updates.json').read_text());sample=np.load(folder/'train_sampled_u2001.npz')
        ref=np.random.default_rng(task['seed']+protocol['training_hard_seed_offset'])
        for k in range(4):
            before=g.digest(ref.bit_generator.state);draw=ref.random((64,len(sample['probability'])));after=g.digest(ref.bit_generator.state)
            for arm in protocol['arms']:
                row=next(r for r in updates if r['arm']==arm and r['update']==2001+k)
                assert row['aux_before_hash']==before and row['aux_after_hash']==after and row['draw_hash']==g.digest(draw)
        expected=ref.random((2,len(sample['probability'])))
        for arm in protocol['arms']:
            z=torch.load(folder/f'{arm}_complete.pt',map_location='cpu',weights_only=False);a=np.random.default_rng();a.bit_generator.state=z['auxiliary_rng']
            assert g.digest(z['auxiliary_rng'])==next(r for r in updates if r['arm']==arm and r['update']==2004)['aux_after_hash']
            np.testing.assert_array_equal(a.random(expected.shape),expected)
        new=torch.load(folder/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs']
        old=torch.load(ROOT/f"results/hard_clean_pilot_v2_20261110/{task['dataset']}_{task['seed']}_HARD_MGDA/evaluation_inputs.pt",map_location='cpu',weights_only=False)['inputs']
        def canonical_edges(inp):
            edges=np.sort(np.asarray(inp['edges']),axis=0)
            order=np.lexsort((edges[1],edges[0]))
            return np.concatenate([edges[:,order],np.asarray(inp['edge_families'])[order][None]],axis=0)
        overlaps=[int(n['t']) for n in new if any(o['t']==n['t'] and np.array_equal(canonical_edges(n),canonical_edges(o)) for o in old)]
        rows.append(dict(dataset=task['dataset'],seed=task['seed'],four_uniform_batches_exact=True,serialized_next_aux_draw_exact=True,new_input_same_Xt_as_old_times=overlaps))
    rt.save(OUT/'state_audit.json',dict(passed=True,rows=rows,all_bound_source_and_history_hashes_unchanged=True,evaluation_script_frozen=True,model_calls=0,GPU_hours=0))
    print(json.dumps(rows))
if __name__=='__main__':main()
