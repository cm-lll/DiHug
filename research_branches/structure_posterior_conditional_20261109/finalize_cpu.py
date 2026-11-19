"""Read-only linear marginal summaries and independent archived-result checks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,json,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/structure_posterior_conditional_20261109'
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def main():
    tick=time.monotonic()
    b=json.loads((OUT/'binding.json').read_text())
    assert hashlib.sha256(Path(b['source']).read_bytes()).hexdigest()==b['source_sha256']
    for p,h in b['assets'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    d=np.load(ROOT/'results/structure_without_edge_ce_theory_20261109/finite_universe.npz')
    bits=d['bits'];q=d['query'].T;fam=d['families'];types=d['types']
    targets=np.load(ROOT/'results/structure_posterior_mcmc_20261109/exact_posteriors.npz')
    tables=np.load(OUT/'conditional_tables.npz')
    edges=json.loads((OUT/'edges.json').read_text());rs=json.loads((OUT/'chains_by_length.json').read_text())
    structures=json.loads((OUT/'structure.json').read_text());cost=json.loads((OUT/'cost.json').read_text())
    linear=[];degree=[];formula_max=0.
    for ci in range(12):
        target=targets[str(ci)];h=tables[str(ci)]
        # Aggregate pairs by their other nine bits, independent of logit calculation.
        for e in range(10):
            for c in np.flatnonzero(bits[:,e]==0):
                other=c+(1<<e);mass=target[c]+target[other]
                value=target[other]/mass
                formula_max=max(formula_max,abs(value-h[c,e]),abs(value-h[other,e]))
        for start in [0,1023,659]:
            for repeat in range(3):
                er=sorted([r for r in edges if r['condition']==ci and r['start']==start and r['repeat']==repeat],key=lambda r:r['edge'])
                truth=np.array([r['exact'] for r in er]);raw=np.array([r['raw'] for r in er]);new=np.array([r['conditional'] for r in er])
                key=dict(condition=ci,retention=er[0]['retention'],observation=er[0]['observation'],start=start,repeat=repeat)
                for relation in range(3):
                    mask=fam==relation;a=float(raw[mask].sum());z=float(new[mask].sum());t=float(truth[mask].sum())
                    linear.append(dict(**key,relation=relation,exact=t,raw=a,conditional=z,raw_error=a-t,conditional_error=z-t,
                        interpretation='sum of per-edge conditional-averaged marginals; linear expected count'))
                for node in range(5):
                    for relation in range(3):
                        mask=(fam==relation)&np.any(q==node,axis=1)
                        if not mask.any():continue
                        a=float(raw[mask].sum());z=float(new[mask].sum());t=float(truth[mask].sum())
                        degree.append(dict(**key,node=node,node_type=int(types[node]),relation=relation,support=int(mask.sum()),
                            exact=t,raw=a,conditional=z,raw_error=a-t,conditional_error=z-t))
    assert formula_max<1e-13
    save('linear_relation_expectations',linear);save('node_expected_relation_degrees',degree)
    aggregate=[]
    for name,rr in [('edge_probability',edges),('linear_relation_count',linear),('node_expected_relation_degree',degree)]:
        for alpha in [0.,.5,.95]:
            v=[r for r in rr if r['retention']==alpha]
            aggregate.append(dict(observable=name,retention=alpha,records=len(v),
                raw_MAE=float(np.mean([abs(r['raw_error']) for r in v])),
                conditional_MAE=float(np.mean([abs(r['conditional_error']) for r in v])),
                worse_records=sum(abs(r['conditional_error'])>abs(r['raw_error'])+1e-12 for r in v)))
    for name in sorted({r['observable'] for r in structures}):
        v=[r for r in structures if r['observable']==name]
        aggregate.append(dict(observable='single_coordinate_graph_observable_'+name,retention=None,records=len(v),
            raw_MAE=float(np.mean([abs(r['raw_error']) for r in v])),
            conditional_MAE=float(np.mean([abs(r['conditional_error']) for r in v])),
            worse_records=sum(abs(r['conditional_error'])>abs(r['raw_error'])+1e-12 for r in v)))
    save('aggregate',aggregate)
    adverse=[dict(kind='edge',**r) for r in edges if abs(r['conditional_error'])>abs(r['raw_error'])+1e-12]
    # Different schemas intentionally stay in separate files.
    save('adverse_edges',adverse)
    save('adverse_structure',[r for r in structures if abs(r['conditional_error'])>abs(r['raw_error'])+1e-12])
    save('verification',dict(status='passed',pair_conditional_max_error=float(formula_max),
        conditional_pairs_checked=12*512*10,old_assets_unchanged=True,
        main_length_chain_improvements=sum(r['conditional_MAE']<r['raw_MAE']-1e-12 for r in rs if r['length']==3500),
        adverse_edge_records=len(adverse),seconds=time.monotonic()-tick,
        RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        limitation='Independent algebra/data check within this session, no external review'))
if __name__=='__main__':main()
