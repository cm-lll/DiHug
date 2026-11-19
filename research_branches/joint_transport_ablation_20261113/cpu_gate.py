"""Predeclared engineering/mechanistic gate, never a final quality guarantee."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from transport import TransportScore,assignment,ROOT
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
audit=loadmod('old_cpu_audit',ROOT/'research_branches/unified_organization_audit_20261113/run.py')
def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False);start=time.time()
    protocol=dict(candidate='transport.py raw_joint_degree_heat_W2_squared_v1',
        gate=['self<1e-12, type permutation<1e-10','finite cost and exact fixed support','mean-count lower bound for all inspected states','controlled reference regular-cycle same-count hub concentration costs more than spread placement','all frozen add/delete/concentrate perturbations have positive cost; >=5pct degree-preserving swaps positive','all reference scales fixed, no selection using final graph quality'],
        decision='One version only. Stop formal branch on scientific gate failure. Historical score trends descriptive only.',
        inputs='120 already frozen CPU perturbations, 2 references; graph fixture cycle8 two extra edges; all seeds and edits reused without reselection')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    p=json.loads((ROOT/'results/paper_ablation_20261112T190000/protocol.json').read_text());scales=json.loads((ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json').read_text());stored=np.load(ROOT/'results/unified_organization_audit_20261113/cpu_perturbation_bits.npz')
    rows=[];checks=[]
    for ds,seed in [('IMDB',2026110411),('PubMed',2026110421)]:
        old=audit.Score(ds,p['parents'][f'{ds}_{seed}_A']['data'],scales)
        capacities=np.bincount(old.f,minlength=old.r);prior=old.count/capacities
        case=dict(space=SimpleNamespace(types=old.types),n_families=old.r,prior=prior)
        score=TransportScore(ds,case,old.q,old.f,old.ref)
        assert score(old.ref)[0]<1e-12
        rng=np.random.default_rng(2026111309);perm=np.arange(old.n)
        for t in np.unique(old.types):ix=np.flatnonzero(old.types==t);perm[ix]=rng.permutation(ix)
        b=old.bits(perm[old.q[:,old.ref]]);pv=score(b)[0];assert pv<1e-10
        for key in stored.files:
            if not key.startswith(ds+'_'):continue
            b=np.unpackbits(stored[key])[:len(old.f)];v=score(b)[0]
            assert np.isfinite(v) and v>1e-12,key
            rows.append(dict(state=key,score=v,parts=score.last))
        (out/(ds+'_scales.json')).write_text(json.dumps(score.s,indent=2))
        checks.append(dict(dataset=ds,self_zero=True,permutation_cost=pv,perturbations=60))
    # Reference regular degree 2 makes degree-only W2 position discrimination
    # unambiguous and independent of vertex identities. Joint test retains heat.
    q=np.array(np.triu_indices(8,1));lookup={tuple(x):i for i,x in enumerate(q.T)};ref=np.zeros(len(q[0]),np.uint8)
    for i in range(8):ref[lookup[tuple(sorted((i,(i+1)%8)))]]=1
    case=dict(space=SimpleNamespace(types=np.zeros(8,int)),n_families=1,prior=np.array([ref.mean()]))
    sc=TransportScore('fixture',case,q,np.zeros(len(ref),int),ref);spread=ref.copy();hub=ref.copy()
    for u,v in [(0,2),(4,6)]:spread[lookup[(u,v)]]=1
    for u,v in [(0,2),(0,4)]:hub[lookup[(u,v)]]=1
    spread_cost=sc(spread)[0];spread_parts=sc.last;hub_cost=sc(hub)[0];hub_parts=sc.last
    assert hub.sum()==spread.sum() and hub_cost>spread_cost+1e-10
    checks.append(dict(fixture='cycle8 same +2 edges',spread_cost=spread_cost,hub_cost=hub_cost,spread_parts=spread_parts,hub_parts=hub_parts))
    (out/'scores.json').write_text(json.dumps(rows,indent=2));(out/'checks.json').write_text(json.dumps(checks,indent=2))
    (out/'receipt.json').write_text(json.dumps(dict(status='passed',seconds=time.time()-start,states=len(rows),models=0,GPU_hours=0,quality_gate=False),indent=2));print((out/'receipt.json').read_text())
if __name__=='__main__':main()
