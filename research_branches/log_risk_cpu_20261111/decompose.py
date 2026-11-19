"""Decompose already-computed CV feedback; no new sampling or model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/log_risk_cpu_20261111/run2'

def main():
    start=time.time();proto=json.loads((OUT/'protocol.json').read_text());rows=[];errs=[]
    for index,item in enumerate(proto['panel']):
        z=np.load(OUT/f'panel_{index:02d}.npz');cv=np.load(OUT/'control_variate'/f'panel_{index:02d}.npz')
        p=np.load(item['path'])['probability'];a=cv['coefficients'];exact=cv['exact_gradient'];gs=[]
        for rep in range(16):
            key=0 if item['kind']=='same_training_input_u2012' else 1
            rng=np.random.default_rng(np.random.SeedSequence([2026111151,item['seed'],key,item['index'],rep]))
            bits=(rng.random((16,len(p)))<p).astype(float)
            c=z['old_costs'][rep*16:(rep+1)*16];f=np.log1p(c)
            f[:,[0,2,3]]-=c[:,[0,2,3]]*a
            g=(f-f.mean(0)).T@(bits-bits.mean(0))/15
            g[[0,2,3]]+=a[:,None]*exact
            err=float(abs(g.mean(0)-cv['new_CV'][rep]).max());errs.append(err)
            np.testing.assert_allclose(g.mean(0),cv['new_CV'][rep],atol=1e-9,rtol=1e-9)
            gs.append(g/5)
        gs=np.asarray(gs);mean=gs.mean(0);var=((gs-mean)**2).sum((0,2))/15
        record={k:v for k,v in item.items() if k!='path'}
        record.update(component_variance_trace=dict(zip(proto['risks'],var.tolist())),
                      component_mean_norm=dict(zip(proto['risks'],np.linalg.norm(mean,axis=1).tolist())),
                      largest_marginal_variance=proto['risks'][int(var.argmax())],
                      warning='Marginal component variances do not sum to total variance without covariance terms')
        rows.append(record)
    (OUT/'control_variate/component_decomposition.json').write_text(json.dumps(rows,indent=2))
    (OUT/'control_variate/decomposition_receipt.json').write_text(json.dumps(dict(seconds=time.time()-start,checks=len(errs),max_abs_error=max(errs),GPU_hours=0,model_calls=0,extra_samples=0),indent=2))
    print(json.dumps(dict(largest_variance_counts={k:sum(r['largest_marginal_variance']==k for r in rows) for k in proto['risks']},seconds=time.time()-start),indent=2))

if __name__=='__main__':main()
