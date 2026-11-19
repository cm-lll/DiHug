"""CPU audit of recorded plans, draws, support, coverage and terminal scoring."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research_branches/cn_multiseed_20261104';SCI=SRC/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
sys.path[:0]=[str(SRC/'experiments/o_cn_matched_execute_v1'),str(SCI/'package')]
from hsgd.bw_preflight import common
from hsgd.kernel import MarginalKernel
from observers import RecordedPlan,digest
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=Path(sys.argv[1]);P=json.loads((out/'protocol.json').read_text());start=time.time();rows=[];probability_rows=[];torch.set_num_threads(1)
    for p,h in P['bindings'].items():assert sha(p)==h,'frozen source/asset changed: '+p
    for ds,m in P['models'].items():
        z=np.load(m['data']);e=z['edges'];e=e.T if e.shape[0]!=2 else e;case=common.make_case(z['types'],e,tuple(map(tuple,z['schema'])));kernel=MarginalKernel(case['prior'],common.schedule(1000))
        fam=np.repeat(np.arange(case['n_families']),case['space'].counts);coverage=np.repeat(case['active'],case['space'].counts).astype(np.uint8)
        for ss in P['sample_seeds']:
            folder=out/ds/str(ss)
            if not (folder/'summary.json').exists():continue
            cache=folder/'cpu_audit.json'
            if cache.exists():
                old=json.loads(cache.read_text())
                for p,h in old['artifact_hashes'].items():assert sha(p)==h,'audited artifact changed: '+p
                rows.append(old['row']);probability_rows.extend(old['probability_rows']);continue
            first_probability_row=len(probability_rows)
            records=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()];assert [r['time'] for r in records]==list(range(1000,0,-1))
            rng=np.random.default_rng(ss);initial=(rng.random(len(fam))<case['prior'][fam]).astype(np.uint8);identity=json.loads((folder/'identity.json').read_text());assert digest(initial)==identity['initial_state_sha256']
            ntrace=0;maxq=0.
            for r in records:
                if time.time()>P['delivery_epoch']-60:raise TimeoutError('delivery reserve')
                t=r['time'];uniform=rng.random(len(fam));assert digest(uniform)==r['uniform_sha']
                plan=RecordedPlan(common.plan_for(case,[ss,t,61],1));ids,f,q=plan.block(0);assert plan.h.hexdigest()==r['query_sha']
                if t in P['times'][str(ss)]:
                    d=np.load(folder/('step_%04d.npz'%t));assert np.array_equal(d['uniform'],uniform) and np.array_equal(d['coverage'],coverage)
                    prob=kernel.reverse(d['p0'][ids],d['xt'][ids],f,t-1,t);maxq=max(maxq,float(abs(prob-d['q'][ids]).max()))
                    assert np.array_equal(d['xprev'][ids],(uniform[ids]<prob).astype(np.uint8))
                    fixed=coverage==0;assert np.array_equal(d['xprev'][fixed],initial[fixed]);ntrace+=1
                    for rel in range(case['n_families']):
                        ix=ids[f==rel]
                        if not len(ix):continue
                        pp=d['p0'][ix];qq=d['q'][ix];chosen=np.where(d['xprev'][ix],qq,1-qq)
                        probability_rows.append(dict(dataset=ds,sample_seed=ss,t=t,relation=rel,slots=len(ix),clean_probability_zero=int((pp==0).sum()),clean_probability_one=int((pp==1).sum()),reverse_probability_zero=int((qq==0).sum()),reverse_probability_one=int((qq==1).sum()),minimum_realized_probability=float(chosen.min()),clean_probability_sum=float(pp.sum()),expected_next_edges=float(qq.sum()),actual_next_edges=int(d['xprev'][ix].sum())))
            grad=torch.load(folder/'score_gradients.pt',map_location='cpu');assert set(grad['times'])==set(P['times'][str(ss)]) and torch.isfinite(grad['gradients']).all()
            final=torch.load(folder/'latest_complete.pt',map_location='cpu');assert final['next_t']==0
            g=np.load(folder/'graph.npz');ge,gf=common.state_edges(case,final['state']);assert np.array_equal(ge,g['edges']) and np.array_equal(gf,g['families'])
            row=dict(dataset=ds,sample_seed=ss,plans_and_draws_verified=1000,traces_verified=ntrace,max_production_q_error=maxq,finite_gradient_elements=grad['gradients'].numel(),graph_sha256=sha(folder/'graph.npz'))
            rows.append(row)
            files=[folder/'graph.npz',folder/'steps.jsonl',folder/'score_gradients.pt',folder/'latest_complete.pt',folder/'identity.json',*folder.glob('step_*.npz')]
            cache.write_text(json.dumps(dict(row=row,probability_rows=probability_rows[first_probability_row:],artifact_hashes={str(p):sha(p) for p in files}),indent=2))
            del grad
    (out/'probability_support.json').write_text(json.dumps(probability_rows,indent=2))
    import csv
    if probability_rows:
        with (out/'probability_support.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(probability_rows[0]));w.writeheader();w.writerows(probability_rows)
    result=dict(rows=rows,seconds=time.time()-start,CPU_only=True,source_asset_hashes_unchanged=True)
    (out/'saved_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
