"""Prepare target specification and finite component-descriptor algebra only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,itertools,json,resource,signal,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/anonymous_teacher_protocol_20261109';START=time.monotonic();HASHES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None:assert h==expected,str(p)
    HASHES[str(p)]=h;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text())
def save(n,o):(OUT/(n+'.json')).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def descriptor(types,labels):
    ts=np.unique(types);counts=np.array([(types==t).sum() for t in ts]);v=np.empty((len(types),len(ts)));hist=[]
    for c in np.unique(labels):
        mask=labels==c;h=np.array([(mask&(types==t)).sum() for t in ts]);v[mask]=h/counts;hist.append(tuple(h))
    return v,tuple(sorted(hist))
def distance(x,y,types):
    values=[]
    for t in np.unique(types):
        mask=types==t;cost=cdist(x[mask],y[mask],metric='cityblock')/x.shape[1];i,j=linear_sum_assignment(cost);values.append(cost[i,j].mean())
    return float(np.mean(values))
def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    save('scope',dict(start_utc=datetime.now(timezone.utc).isoformat(),
        work='Read actual six-term scoring and raw reference components; define one proposed anonymous component discrepancy; deterministic partition algebra, no real generated-graph new score',
        budget=dict(threads=1,seconds=120,memory_MiB=1024,GPU=0),
        not_done=['teacher sampling','new energy selection','tolerance fitting','network','training','generation','production edit']))
    prior=read(ROOT/'results/real_structure_envelope_20261109/binding.json')['assets']
    pkg=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package'
    cfg=read(pkg/'config.json');records=[]
    for ds in ['IMDB','PubMed']:
        p=pkg/'assets'/(ds+'.npz')
        with np.load(bind(p,prior[str(p)])) as z:
            types=z['types'];schema=z['schema'];edges=z['edges'];edges=edges if edges.shape[0]==2 else edges.T
        a=np.zeros((len(types),len(types)),bool);a[edges[0],edges[1]]=a[edges[1],edges[0]]=True
        labels=np.full(len(types),-1);cid=0
        for u in range(len(types)):
            if labels[u]>=0:continue
            stack=[u];labels[u]=cid
            while stack:
                v=stack.pop();new=np.flatnonzero(a[v]&(labels<0));labels[new]=cid;stack.extend(new)
            cid+=1
        x,hist=descriptor(types,labels);assert distance(x,x,types)==0
        nt=np.bincount(types);components=[]
        for h in hist:
            components.append(dict(size=int(sum(h)),type_counts=list(map(int,h)),type_equal_mass=float(np.mean(np.array(h)/nt))))
        records.append(dict(dataset=ds,nodes=len(types),type_names=cfg['datasets'][ds]['type_names'],type_counts=nt.tolist(),
            components=components,reference_self_distance=0.,
            configuration_warning='Package config read only for type-name metadata; its historical normal2 training settings are NOT the current normal1 protocol.'))
    save('reference_components',records)
    # Enumerate all 15 set partitions of four nodes; no random graph construction.
    types=np.array([0,0,1,1]);partitions={}
    for raw in itertools.product(range(4),repeat=4):
        mapping={};lab=[]
        for c in raw:
            if c not in mapping:mapping[c]=len(mapping)
            lab.append(mapping[c])
        partitions[tuple(lab)]=descriptor(types,np.array(lab))
    assert len(partitions)==15
    items=list(partitions.items());D=np.zeros((15,15));zero_equiv=0;closed_errors=[]
    for i,(lab,(x,hist)) in enumerate(items):
        for j,(_, (y,hist2)) in enumerate(items):
            D[i,j]=distance(x,y,types)
            assert (D[i,j]<1e-12)==(hist==hist2);zero_equiv+=1
        w=[np.mean(np.array(h)/np.bincount(types)) for h in hist]
        closed=1-sum(v*v for v in w)
        closed_errors.append(abs(distance(x,np.ones_like(x),types)-closed))
    trierr=max(float(D[i,k]-D[i,j]-D[j,k]) for i,j,k in itertools.product(range(15),repeat=3))
    assert abs(D-D.T).max()<1e-12 and max(closed_errors)<1e-12 and trierr<1e-12
    save('finite_checks',dict(partitions=15,pair_checks=zero_equiv,triangle_inequality_checks=15**3,
        symmetry_error=float(abs(D-D.T).max()),triangle_inequality_max_violation=max(0,trierr),connected_closed_form_max_error=max(closed_errors),
        zero_iff_same_component_type_histogram=True,scope='Finite descriptors, not proof of general graph identifiability, training success or real-time cost'))
    sources=['research_branches/explicit_structure_cpu_20261109/explicit_score.py',
        'research_branches/joint_degree_distance_20261109/run.py',
        'research_branches/structure_posterior_real_cost_20261109/run_cpu.py',
        'research_branches/structure_sparse_target_20261109/run_cpu.py']
    for f in sources:bind(ROOT/f)
    oldcost=read(ROOT/'results/structure_posterior_real_cost_20261109/budget_extrapolation.json')
    inventory=read(ROOT/'results/real_structure_envelope_20261109/asset_inventory.json')
    # Fixed next-proposal panel chosen by identity, no quality ranking.
    panel=[j for j in inventory if j['panel']=='matched_four_grid' and j['arm']=='A' and j['sample_seed']==2026102821]
    assert len(panel)==4
    proposal=dict(status='proposed_not_executed',graphs='2 references + 4 archived A graphs = 6 fixed backgrounds',
        archived_graphs=[{k:j[k] for k in ['dataset','train_seed','sample_seed','file','expected_sha']} for j in panel],
        operations='Per background/relation first 4 existing and first 4 absent mutable slots under fixed SHA order, no replacement if insufficient; at most 480 independent flips. Not a Markov chain.',
        checks=['reference identity and type-permutation invariance','component delta vs full BFS/assignment','joint/closure/component changes reported separately without total reward or selected energy',
            'component-neutral within-component changes retained, split/merge changes tagged','fixed support and no self-loop checks','actual timing, not old-cost guarantee'],
        budget=dict(recommended_wall_minutes=10,CPU_threads=1,memory_GiB=2,GPU_hours=0),
        old_480_query_seconds=480*oldcost['incremental_seconds_per_flip'],
        uncertainty='Old .841ms is a different six-term score, excludes new component recomputation and sequential commit; 10min is an envelope, not measured throughput.',
        forbidden=['posterior teacher chain','new network call','training','new full diffusion graph','tolerance or temperature search'])
    save('next_cpu_proposal',proposal)
    save('target_specification',dict(status='definitions_proposed_not_production_frozen',
        graph_support='Fixed typed inventory, legal unordered simple slots, original per-relation zero/one support',
        base='Independent Bernoulli original reference relation frequencies, no exact count quota',
        existing_terms=['joint_relation_degree_transport','relation_degree_W1','low_rank_error','hub_rank_error','typed_triangle_L1','node_triangle_W1'],
        added_proposed_term='Typewise W1 of per-node component type-count / full type inventory vector, L1 cost / number of types; equal type weights',
        target_family='min KL(P||mu) with explicit per-term expectation bounds; any tail requirements separately declared; no current numerical bounds selected',
        tolerances=None,tail_probability=None,tail_severity=None,multipliers=None,
        teacher='rho(G|Xt,t) proportional rho0(G)*q_t(Xt|G); labels are posterior marginal clean-edge probabilities, not full conditional at arbitrary hard background',
        training_coverage='Must separately specify draws from rho0 and q_t or valid importance/coverage scheme; original-reference-only corruption is not automatically equivalent',
        inference='No reference graph scoring, no quota, no online edit repair; original network/kernel interface maintained only if later validated',
        missing=['real-scale tolerance rule','target solver/normalizer or approximate constrained inference','sequential cache commit correctness','posterior mixing and label accuracy','training input coverage','parameter/gradient cost and complete generation validation']))
    assert all(sha(p)==h for p,h in HASHES.items())
    save('binding',dict(source=str(Path(__file__)),source_sha256=sha(Path(__file__)),assets=HASHES))
    save('receipt',dict(status='complete_stopped',seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        GPU_hours=0,model_calls=0,training_updates=0,new_random_graphs=0,new_real_graph_scores=0,old_assets_unchanged=True,formal_training_authorized=False))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
