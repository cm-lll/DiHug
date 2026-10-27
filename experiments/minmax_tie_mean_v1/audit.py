"""Independent CPU inspection of saved evidence, without model forward/update."""
import argparse,json,pathlib,hashlib,numpy as np,torch
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',required=True);ap.add_argument('--capture',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();r=pathlib.Path(a.evidence)
    cap=torch.load(a.capture,map_location='cpu',weights_only=False);contract=json.loads((r/'contract.json').read_text());atol=contract['atol'];rtol=contract['rtol'];report={}
    assert sha(pathlib.Path(a.capture))==json.loads((r/'capture_binding.json').read_text())['sha256']
    for name,c in cap.items():
        x=c['src'].numpy();b=c['batch'].numpy();up=c['upstream'].numpy();y=c['output'].numpy();mask=x==y[b];counts=np.stack([mask[b==k].sum(0) for k in range(len(y))]);expected=mask*(up[b]/counts[b]);z=np.load(r/(name+'_gpu20.npz'))
        conservation=[]
        for i in range(20):
            out=z[str(i)+'_output'];g=z[str(i)+'_gradient'];assert np.array_equal(out,y);assert out.tobytes()==z['0_output'].tobytes();assert g.tobytes()==z['0_gradient'].tobytes();np.testing.assert_allclose(g,expected,atol=atol,rtol=rtol);assert not (g[~mask]!=0).any()
            totals=np.stack([g[b==k].astype(np.float64).sum(0) for k in range(len(y))]);np.testing.assert_allclose(totals,up,atol=atol,rtol=rtol);conservation.append(float(np.max(np.abs(totals-up))))
        report[name]=dict(audited_repeats=20,forward_exact=True,gradient_repeat_exact=True,formula=True,max_conservation_error=max(conservation))
    summary=json.loads((r/'summary.json').read_text());assert len(summary['cells'])==6
    for label,row in summary['cells'].items():
        folder=r/label;state=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)['model'];manifest=json.loads((folder/'initial_tensor_hashes.json').read_text())
        assert sha(folder/'initial.pt')==row['initial_sha256']
        for name,v in state.items():assert hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()==manifest[name]['sha256']
        same=json.loads((folder/'same_gpu.json').read_text());assert same['passes']==row['same_gpu_passed'];assert all(v['passes'] for v in same['gradients'].values())==row['same_gpu_passed']
        if row['updates']:
            assert summary['operator'] and all(v['same_gpu_passed'] for v in summary['cells'].values());assert row['updates']<=3
            if (folder/'after1.pt').exists():
                assert sha(folder/'after1.pt')==row['checkpoint_sha256']
                checkpoint=torch.load(folder/'after1.pt',map_location='cpu',weights_only=False)
                assert checkpoint['updates']==1 and checkpoint['binding']['initial_sha256']==row['initial_sha256']
                assert all(float(v['step'])==1 for v in checkpoint['optimizer']['state'].values())
            restore=json.loads((folder/'restore.json').read_text());assert restore['passes']==row['restore_passed'];assert all(row['load_exact'].values())
        report[label]=dict(initial_tensor_hashes=True,same_gpu_passes=row['same_gpu_passed'],actual_updates=row['updates'],restore_passes=row.get('restore_passed'))
    assert sum(v['updates'] for v in summary['cells'].values())==summary['optimizer_step_calls']<=18
    report.update(status='passed',scientific_training=0,generated_graphs=0,optimizer_step_calls=summary['optimizer_step_calls'])
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
