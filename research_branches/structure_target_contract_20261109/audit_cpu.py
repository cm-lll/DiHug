"""Read-only saved-score frontier and sparse-prior contract audit. No new graphs."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,resource,signal,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structure_target_contract_20261109'
START=time.monotonic();BIND={}
def read(p):
    p=ROOT/p;BIND[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(n,v):
    (OUT/(n+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(v,list) and v:
        with (OUT/(n+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(v[0]));w.writeheader();w.writerows(v)
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    keys=['joint','relation_W1','low','hub','typed_triangle_L1','node_triangle_W1','component']
    save('protocol',dict(start_utc=datetime.now(timezone.utc).isoformat(),work='Saved-score Pareto audit and contract design, no rescoring or target fitting',
        grouping='within identical background; stricter endpoint comparison within same relation and operation',
        numerical_epsilon=1e-10,epsilon_mean=None,epsilon_graph=None,tail_probability=None,tail_severity=None,
        safeguards='No tolerance inferred from bad generated graphs, sampled edit shell, or numerical tolerance. No new lambda/temperature.',
        limits=dict(seconds=120,CPU_threads=1,memory_GiB=1,GPU_hours=0)))
    rows=read('results/component_target_cpu_20261109/verified/flips.json')
    bases=read('results/component_target_cpu_20261109/verified/backgrounds.json')
    old_binding=read('results/component_target_cpu_20261109/verified/binding.json')
    for p,h in old_binding.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    records=[];summary=[]
    for base in bases:
        group=[r for r in rows if (r['dataset'],r['background'])==(base['dataset'],base['background'])]
        D=np.array([[r[k+'_delta'] for k in keys] for r in group]);E=1e-10
        # An accepted budget would be on absolute errors, not just their improvements.
        initial=np.array([base['component_distance'] if k=='component' else base[k] for k in keys])
        assert (initial+D>=-E).all()
        costfree=(D[:,:6]<=E).all(1)&(D[:,6]<-E)
        improved=D[:,6]<-E;details=[]
        for i,r in enumerate(group):
            if not improved[i]:continue
            weak=(D<=D[i]+E).all(1);strict=(D<D[i]-E).any(1)
            same=np.array([v['relation']==r['relation'] and v['operation']==r['operation'] for v in group])
            better=np.flatnonzero(weak&strict&same)
            records.append(dict(dataset=r['dataset'],background=r['background'],relation=r['relation'],u=r['u'],v=r['v'],
                component_delta=r['component_delta'],no_six_term_cost=bool(costfree[i]),
                worsened_terms=[keys[k] for k in np.flatnonzero(D[i,:6]>E)],
                same_relation_operation_dominators=[dict(u=group[j]['u'],v=group[j]['v']) for j in better],
                required_absolute_budget={k:float(initial[j]+D[i,j]) for j,k in enumerate(keys)}))
            details.append(len(better)>0)
        summary.append(dict(dataset=base['dataset'],background=base['background'],tested=len(group),component_improving=int(improved.sum()),
            component_improving_without_other_cost=int(costfree.sum()),with_cost=int((improved&~costfree).sum()),
            component_improving_dominated_same_relation_operation=sum(details),
            best_component_delta_with_no_other_cost=None if not costfree.any() else float(D[costfree,6].min()),
            warning='Pareto analysis of fixed small archived selection, not all legal edges, probabilities or a training rule'))
    save('bridge_frontier',records);save('frontier_summary',summary)
    # Reuse exact old distributions; do not solve another target or choose tolerances.
    paths=['results/structure_target_constraints_20261109/completed/targets.npz','results/structure_without_edge_ce_theory_20261109/finite_universe.npz']
    for p in paths:BIND[str(ROOT/p)]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    with np.load(ROOT/paths[0]) as z:
        codes=z['codes'];mu=z['sparse_base'];weighted=z['bounded_expected_error'];F=z['features'];eps=z['bounds']
    with np.load(ROOT/paths[1]) as u:
        bits=u['bits'][codes];families=u['families'];reference=u['reference']
    count=np.stack([bits[:,families==r].sum(1) for r in np.unique(families)],1)
    target=np.array([reference[families==r].sum() for r in np.unique(families)])
    count_rows=[]
    for r in range(count.shape[1]):
        count_rows.append(dict(relation=r,reference_count=int(target[r]),base_expected_count=float(mu@count[:,r]),
            structure_tilt_expected_count=float(weighted@count[:,r]),shift=float(weighted@count[:,r]-target[r])))
    assert np.max(abs(mu@count-target))<1e-10
    # Exact local sensitivity of count means to linear count penalties: -covariance.
    logs=np.log(weighted);centered=count-weighted@count;cov=centered.T@(weighted[:,None]*centered)
    def means(h):
        z=logs-count@h;z-=z.max();p=np.exp(z);p/=p.sum();return p@count
    delta=1e-5;fd=np.column_stack([(means(np.eye(count.shape[1])[j]*delta)-means(-np.eye(count.shape[1])[j]*delta))/(2*delta) for j in range(count.shape[1])])
    err=float(abs(fd+cov).max());assert err<1e-8
    save('sparse_baseline_check',dict(relations=count_rows,max_covariance_derivative_error=err,
        old_mean_constraints_satisfied=bool(np.all(weighted@F<=eps+1e-7)),
        old_per_graph_joint_pass_probability=float(weighted@np.all(F<=eps+1e-12,axis=1)),
        limitation='Archived five-node example; not measured IMDB/PubMed effect, not a new optimized target'))
    save('contract',dict(status='proposed_not_training_ready',
        support='fixed type inventory/schema/simple undirected slots/reference zero-one relation support',
        base='reference relation Bernoulli frequencies',
        proposed_density_guard='E_rho[M_r] = M_r(reference) for mutable relations; fixed relations removed from redundant equality system',
        density_boundary='population mean only; not per-graph edge quota, not proof of count variance control',
        objectives=keys,mean_bounds=None,per_graph_bounds=None,tail_probability=None,tail_severity=None,
        no_type_masking='Before real freeze export existing per-type terms. Whole-type mean does not protect Chemical/hub or triangle subclasses.',
        target='min KL(rho||mu) subject to declared mean, union-tail/severity, density constraints; eta_r unrestricted, lambda_j nonnegative',
        training='sample clean states from rho and corrupt with original q, or prove a valid coverage/importance scheme; soft labels are posterior marginals',
        inference='network plus original reverse interface; no deployed reference scoring/edit repair',
        not_claimed=['new score is a generator','sparse base alone fixes density','expected bounds guarantee every graph','min-KL guarantees nonisomorphic diversity','teacher can be estimated cheaply','production reverse joint law is exact'],
        actual_bounds_source='Must be a predeclared scientific tolerance policy. One reference graph does not identify population variability.',
        next='Reference-only tolerance calibration design, separating engineering edit budget from uncertainty; no new teacher or model until bounded proposal approved'))
    for p,h in BIND.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    save('binding',dict(assets=BIND,source=str(Path(__file__)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        GPU_hours=0,model_calls=0,training_updates=0,new_graphs=0,new_score_evaluations=0,new_target_fits=0,assets_unchanged=True,formal_training_authorized=False))
    print(json.dumps(dict(summary=summary,sparse=json.loads((OUT/'sparse_baseline_check.json').read_text()),receipt=json.loads((OUT/'receipt.json').read_text())),ensure_ascii=False))
if __name__=='__main__':main()
