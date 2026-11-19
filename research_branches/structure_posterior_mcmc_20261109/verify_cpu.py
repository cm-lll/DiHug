"""Read saved toy chains; independent probability, RNG and accounting checks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, hashlib, itertools, json, resource, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/structure_posterior_mcmc_20261109'
OLD=ROOT/'results/structure_without_edge_ce_theory_20261109'
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)
def main():
    tick=time.monotonic();p=json.loads((OUT/'protocol.json').read_text())
    bind=json.loads((OUT/'binding.json').read_text())
    for f,h in bind['assets'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(bind['source']).read_bytes()).hexdigest()==bind['source_sha256']
    data=np.load(OLD/'finite_universe.npz');bits=data['bits'];families=data['families'];edges=data['query'].T
    old=json.loads((OLD/'exact_posterior_distributions.json').read_text())
    target_archive=np.load(OUT/'exact_posteriors.npz');paths=np.load(OUT/'trajectories.npz')
    rows=json.loads((OUT/'chains.json').read_text()); edge_rows=json.loads((OUT/'edge_marginals.json').read_text())
    aa=json.loads((OUT/'acceptance.json').read_text())
    assert len(rows)==108 and len(edge_rows)==1080 and len(paths.files)==108
    assert sum(r['attempts'] for r in aa)==432000
    assert all(r['attempts']==r['null']+r['accepted_state_changes']+r['rejected'] for r in aa)
    # Physical observables evaluated as expectations under target and retained chain.
    lookup={tuple(e):i for i,e in enumerate(edges)}
    triangles=np.zeros(1024)
    for a,b,c in itertools.combinations(range(5),3):
        triangles+=np.prod(bits[:,[lookup[(a,b)],lookup[(a,c)],lookup[(b,c)]]],axis=1)
    components=[]
    for b in bits:
        groups=[{i} for i in range(5)]
        for i,(u,v) in enumerate(edges):
            if not b[i]:continue
            gu=next(g for g in groups if u in g);gv=next(g for g in groups if v in g)
            if gu is not gv:gu.update(gv);groups.remove(gv)
        components.append(len(groups))
    observable={f'relation_{r}_edge_count':bits[:,families==r].sum(1) for r in range(3)}
    observable.update(triangle_count=triangles,component_count=np.array(components),energy=data['energy'])
    for k,name in enumerate(['number_error','joint_allocation_error','closure_error','connection_error']):
        observable[name]=data['energy_parts'][:,k]
    structure=[]; rng_checks=0; max_target_diff=0.
    for row in rows:
        ci=row['condition'];alpha=row['retention'];obs=row['observation'];x=bits[obs]
        prob=alpha*bits+(1-alpha)*np.array([.25,.4,.35])[families]
        post=data['illustrative_gibbs']*np.prod(np.where(x,prob,1-prob),axis=1);post/=post.sum()
        original=next(r for r in old if r['prior']=='illustrative_gibbs' and r['retention']==alpha and r['observation_code']==obs)
        diff=max(abs(post-target_archive[str(ci)]).max(),abs(post-np.array(original['posterior'])).max())
        max_target_diff=max(max_target_diff,float(diff));assert diff<1e-13
        si=p['starts'].index(row['start']);rep=row['repeat'];path=paths[f'c{ci}_s{si}_r{rep}'];assert len(path)==4000
        # Independently replay first 32 proposal choices using masks, not stored transition tables.
        rng=np.random.default_rng(row['seed']);c=row['start']
        for step in range(32):
            k=int(np.searchsorted([.5,.75,1.],rng.random(),side='right'))
            ts=p['proposal_templates'][['toggle','relocate','switch'][k]]
            mask,left,right=ts[int(rng.integers(len(ts)))];u=rng.random()
            proposed=c^mask if k==0 or c&mask in (left,right) else c
            if proposed!=c and u<min(1.,post[proposed]/post[c]):c=proposed
            assert c==path[step];rng_checks+=1
        freq=np.bincount(path[500:],minlength=1024)/3500
        assert abs(abs(freq-post).sum()/2-row['TV'])<1e-12
        for name,values in observable.items():
            exact=float(post@values);est=float(freq@values)
            structure.append(dict(condition=ci,retention=alpha,observation=obs,start=row['start'],repeat=rep,
                                  observable=name,exact_expectation=exact,estimate=est,signed_error=est-exact))
    summaries=[]
    for alpha in p['retentions']:
        r=[v for v in rows if v['retention']==alpha]
        record=dict(retention=alpha,chains=len(r))
        for name in ['edge_MAE','edge_max_error','TV','iid_edge_MAE','iid_TV','energy_ESS','edge_count_ESS']:
            vals=np.array([v[name] for v in r if v[name] is not None])
            record[name+'_mean']=float(vals.mean());record[name+'_min']=float(vals.min());record[name+'_max']=float(vals.max())
        summaries.append(record)
    # Average each start's three runs only for a descriptive start-sensitivity table.
    start_rows=[]
    for ci in range(12):
        means=[]
        for start in p['starts']:
            rs=[r for r in edge_rows if r['condition']==ci and r['start']==start]
            means.append(np.array([np.mean([r['estimate'] for r in rs if r['edge']==e]) for e in range(10)]))
        distances=[float(abs(means[i]-means[j]).mean()) for i,j in itertools.combinations(range(3),2)]
        start_rows.append(dict(condition=ci,mean_edge_difference_between_start_means=float(np.mean(distances)),
                               max_edge_difference_between_start_means=max(distances),
                               interpretation='Includes Monte Carlo fluctuation; exact-distribution start bias reported separately'))
    save('summary',summaries);save('structure_expectations',structure);save('start_sensitivity',start_rows)
    save('verification',dict(status='passed',raw_rng_steps_checked=rng_checks,chains_accounted=108,
        max_posterior_difference_previous_exact=max_target_diff,unchanged_assets=True,
        seconds=time.monotonic()-tick,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        limitation='CPU independent formulation and saved-data checks, not independent researcher review'))
if __name__=='__main__':main()
