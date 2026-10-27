"""Independent CPU array audit; no network, no autograd, no optimizer."""
import argparse,json,pathlib
import numpy as np
import torch
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);a=ap.parse_args();root=pathlib.Path(a.results)
    captured=torch.load(root/'captured.pt',map_location='cpu',weights_only=False);report={}
    for name,cap in captured.items():
        src=cap['src'].numpy();batch=cap['batch'].numpy();g=cap['upstream'].numpy();out=cap['output'].numpy()
        assert np.array_equal(out,np.stack([src[batch==b].max(0) for b in range(len(out))]))
        mask=src==out[batch];multiplicity=np.stack([mask[batch==b].sum(0) for b in range(len(out))]);unique=multiplicity==1
        with np.load(root/(name+'_gpu20.npz')) as z:
            for i in range(20):
                value,idx,grad=[z[str(i)+'_'+key] for key in ['output','index','gradient']]
                assert np.array_equal(value,out)
                assert (idx>=0).all() and (idx<len(src)).all()
                expected=np.zeros_like(src)
                for b in range(len(out)):
                    assert np.array_equal(batch[idx[b]],np.full(src.shape[1],b))
                    assert np.array_equal(src[idx[b],np.arange(src.shape[1])],out[b])
                    expected[idx[b],np.arange(src.shape[1])]=g[b]
                assert np.array_equal(grad,expected)
                assert np.array_equal(np.stack([grad[batch==b].sum(0) for b in range(len(out))]),g)
                assert not ((grad!=0)&~mask).any()
        with np.load(root/(name+'_cpu.npz')) as z:
            expected_cpu=np.where(mask,g[batch]/multiplicity[batch].astype(np.float32),np.float32(0))
            assert np.array_equal(z['output'],out)
            np.testing.assert_allclose(z['gradient'],expected_cpu,atol=2e-6,rtol=1e-5)
            cpu_diff=float(np.abs(z['gradient']-expected_cpu).max())
        report[name]=dict(repeats_audited=20,gpu_gradient_exact_single_argmax_scatter=True,gradient_conservation_exact=True,
            cpu_equal_tie_split_max_error=cpu_diff,unique_channels=int(unique.sum()),tied_channels=int((multiplicity>1).sum()),
            unique_case_tested=bool(unique.any()))
    # Includes all observer retries; compare content hashes, not serialization bytes.
    hashes=json.loads((root/'state_tensor_hashes.json').read_text())
    for attempt in ['attempt1','attempt2']:
        old=json.loads((root/attempt/'state_tensor_hashes.json').read_text());assert old==hashes
    report['all_attempt_reconstructions_same_content']=True
    report['actual_network_forwards']=sum(json.loads(p.read_text())['model_forward_calls'] for p in [root/'summary.json',root/'attempt1/summary.json',root/'attempt2/summary.json'])
    report['actual_network_backwards']=report['actual_network_forwards'];report['optimizer_steps']=0
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
