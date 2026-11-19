"""CPU independent estimator and saved-record checks; no model imports."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import itertools,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/one_step_terminal_gpu_preflight_20261111'

def main():
    p=np.array([.1,.35,.8]); bits=np.array(list(itertools.product([0,1],repeat=3)),dtype=float)
    prob=np.prod(np.where(bits,p,1-p),axis=1)
    cost=(bits.sum(1)-1)**2+.4*bits[:,0]*bits[:,1]
    exact=(prob*cost)@(bits-p)
    # K=1 no-baseline is unbiased; pairwise LOO integrates to the same derivative.
    pair=np.zeros(3)
    for i in range(8):
        for j in range(8):
            pair+=prob[i]*prob[j]*(cost[i]-cost[j])*(bits[i]-bits[j])/2
    np.testing.assert_allclose(pair,exact,atol=1e-14)
    logits=np.log(p/(1-p)); fd=[]
    def value(z):
        q=1/(1+np.exp(-z));return np.prod(np.where(bits,q,1-q),axis=1)@cost
    for j in range(3):
        e=np.zeros(3);e[j]=1e-5;fd.append((value(logits+e)-value(logits-e))/2e-5)
    np.testing.assert_allclose(fd,exact,atol=1e-9)
    rows=json.loads((OUT/'rows.json').read_text()); complete=json.loads((OUT/'complete.json').read_text())
    assert len(rows)==64 and complete['VJPs']==192 and complete['updates']==0
    assert len({(r['dataset'],r['sample_seed'],r['t']) for r in rows})==64
    for r in rows:
        c=np.array(r['per_graph_costs']);a=np.array(r['per_graph_LOO_cost_advantages'])
        independent=np.array([c[i]-np.delete(c,i).mean() for i in range(16)])
        np.testing.assert_allclose(a,independent,atol=1e-12)
        assert abs(a.sum())<1e-10
        assert len(r['type_joint_costs'])==16 and all(len(x)==4 for x in r['type_joint_costs'])
        assert all(np.isfinite(r['gradient_norms'])) and min(r['gradient_norms'])>=0
        assert r['active_slots']==(13120 if r['dataset']=='IMDB' else 18336)
    for ds in ['IMDB','PubMed']:
        x=json.loads((OUT/(ds+'_terminal_costs.json')).read_text());c=np.array(x['costs'])
        np.testing.assert_allclose(x['LOO_cost_advantages'],[c[i]-np.delete(c,i).mean() for i in range(4)],atol=1e-12)
    for p,h in json.loads((OUT/'protocol.json').read_text())['bindings'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    result=dict(status='pass',exact_S1_and_LOO_expectation=True,finite_difference_max_abs=float(np.max(np.abs(exact-fd))),
        saved_weight_checks=64,typed_support_checks=64,terminal_weight_checks=2,bindings_unchanged=True,
        limitations='No unbiasedness claim for terminal eager surrogate; no parameter updates, optimizer restoration or quality test.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
