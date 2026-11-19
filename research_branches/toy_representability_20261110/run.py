"""Finite CPU feature-orbit/conditional-output audit; no training or new paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,json,csv,hashlib,itertools,resource,signal,warnings,importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/toy_representability_20261110';PARENT=ROOT/'results/mean_anchored_toy_20261110_attempt2'
s=importlib.util.spec_from_file_location('toy',ROOT/'research_branches/full100_toy_20261110/run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,rows):
    with (OUT/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def partition(F,edges):
    _,ix=np.unique(F[edges],axis=0,return_inverse=True)
    return [edges[ix==j] for j in range(ix.max()+1)]
def bound(groups,target):
    # All observed partitions are proven polynomial cases below. Anything else stops.
    shape=tuple(sorted((len(g),int(target[g].sum())) for g in groups))
    if all(k in (0,n) for n,k in shape):return 1.,'pure_groups',shape
    if shape==((4,2),):return .125,'2*p^2*(1-p)^2 <= 1/8',shape
    if shape==((2,1),(2,1)):return .125,'2*p*(1-p)*q*(1-q) <= 1/8',shape
    if shape==((1,0),(1,1),(2,1)):return .25,'p*(1-p)*(q*(1-r)+(1-q)*r) <= 1/4',shape
    raise ValueError(('unproved partition',shape))
def probability_good(p,good):return float(np.prod(np.where(good>0,p,1-p),axis=1).sum())
def main():
    start=time.time();OUT.mkdir(exist_ok=True);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU limit')));signal.alarm(540)
    bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('watch.py'),ROOT/'research_branches/full100_toy_20261110/run.py',PARENT/'protocol.json',PARENT/'audit.json',PARENT/'state_index.json',PARENT/'states_and_paths.npz',PARENT/'finals.json']}
    f,C,tri,cfg,kernel=r.setup();bindings.update(r.BIND);assert json.loads((PARENT/'audit.json').read_text())['passed']
    good=np.flatnonzero(abs(C).max(1)<1e-14);assert len(good)==2;Bgood=f.B[good];cross=np.flatnonzero(f.f==1);edge_lookup={tuple(e):i for i,e in enumerate(f.edges)}
    protocol=dict(scope='all 64 existing states, type-preserving automorphisms, per-state linear separation; archived initial and update100 raw/EMA exact t1 occupancies',limits=dict(CPU_threads=1,RSS_MiB=2048,science_seconds=540,watch_seconds=600,GPU=0),models='4-node 9-parameter toy only',new_training=0,new_sampled_paths=0,states=64,linear_programs=128,archived_readouts=21,good_state_ids=good.tolist(),no_rule_tuning=True,bindings=bindings)
    save('protocol.json',protocol);ph=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();save('freeze.json',dict(sha256=ph,epoch=time.time()))
    perms=[np.array(a+b) for a in itertools.permutations([0,1]) for b in itertools.permutations([2,3])];state_rows=[];lp_rows=[];detail=[];caps=[];auto_caps=[];errors=[]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected')
        for x,F in enumerate(f.features[1]):
            groups=partition(F,cross);cap,formula,shape=bound(groups,Bgood[0]);caps.append(cap)
            autos=[perm for perm in perms if np.array_equal(f.adj[x],f.adj[x][np.ix_(perm,perm)])]
            unseen=set(map(int,cross));orbits=[]
            while unseen:
                e=min(unseen);i,j=f.edges[e];orb={edge_lookup[tuple(sorted((int(p[i]),int(p[j]))))] for p in autos};unseen-=orb;orbits.append(np.array(sorted(orb)))
            acap,aformula,ashape=bound(orbits,Bgood[0]);auto_caps.append(acap);assert acap>=cap
            row=dict(state=x,edge_count=int(f.B[x].sum()),feature_groups=len(groups),automorphism_groups=len(orbits),feature_good_probability_cap=cap,equivariant_feature_cap=acap,extra_feature_loss=bool(cap<acap),features_partition=str(shape),automorphism_partition=str(ashape));state_rows.append(row)
            detail.append(dict(**row,feature_edges=[g.tolist() for g in groups],automorphism_edges=[g.tolist() for g in orbits],formula=formula,automorphism_formula=aformula))
            # Construct a maximizing distribution for each observed partition by a
            # small exact grid containing all analytic maximizers (0,1/2,1).
            best=-1.;bestp=None
            for choice in itertools.product([0.,.5,1.],repeat=len(groups)):
                p=np.ones(f.M)
                for g,v in zip(groups,choice):p[g]=v
                score=probability_good(p,Bgood)
                if score>best:best,bestp=score,p.copy()
            assert abs(best-cap)<1e-14;detail[-1]['attaining_probabilities']=bestp.tolist()
            feasible=[]
            for target in good:
                y=2*f.B[target]-1
                lp=linprog(np.zeros(f.D),A_ub=-(y[:,None]*F),b_ub=-np.ones(f.M),bounds=[(None,None)]*f.D,method='highs',options={'threads':1})
                if lp.status not in (0,2):raise RuntimeError(('LP unresolved',x,int(target),lp.message))
                margin=float(np.min(y*(F@lp.x))) if lp.success else None
                if lp.success:assert margin>=1-1e-7
                feasible.append(lp.success);lp_rows.append(dict(state=x,target=int(target),feasible=bool(lp.success),solver_status=int(lp.status),margin=margin,theta=lp.x.tolist() if lp.success else None))
            row['one_target_linearly_separable']=any(feasible)
            if any(feasible):assert cap==1
    table('state_bounds.csv',state_rows);save('state_details.json',detail);save('linear_separation.json',lp_rows)
    bank=np.load(PARENT/'states_and_paths.npz');index=json.loads((PARENT/'state_index.json').read_text());finals=json.loads((PARENT/'finals.json').read_text());selected=[(i,x) for i,x in enumerate(index) if x['updates'] in (0,100)];assert len(selected)==21
    rows=[];state_mass=[];caps=np.array(caps);auto_caps=np.array(auto_caps)
    for j,meta in selected:
        theta=bank['parameters'][j];dist=f.prior.copy()
        for t in range(100,1,-1):dist=dist@r.transitions(f,theta,t)[0]
        p=expit(f.features[1]@theta);one=np.array([probability_good(q,Bgood) for q in p]);actual=float(dist@one);bound_now=float(dist@caps);auto_now=float(dist@auto_caps)
        terminal=dist@r.transitions(f,theta,1)[0];err=float(abs(terminal-bank['distributions'][j]).max());assert err<1e-12;errors.append(err)
        assert np.all(one<=caps+1e-12) and actual<=bound_now+1e-12
        case=dict(**meta,actual_good_probability=actual,same_X1_feature_upper_bound=bound_now,same_X1_automorphism_upper_bound=auto_now,mass_on_feature_limited_states=float(dist[caps<1].sum()),mass_on_extra_feature_loss_states=float(dist[caps<auto_caps].sum()),mass_on_automorphism_limited_states=float(dist[auto_caps<1].sum()),feature_gap=float(auto_now-bound_now),remaining_relaxation_gap=float(bound_now-actual),terminal_error=err)
        rows.append(case)
        for x in range(f.S):state_mass.append(dict(**meta,state=x,mass_X1=float(dist[x]),conditional_good_probability=float(one[x]),feature_cap=float(caps[x]),automorphism_cap=float(auto_caps[x])))
    table('readout_bounds.csv',rows);table('X1_state_details.csv',state_mass)
    # Explicit symmetric-state example: independent marginals versus correlated
    # distribution with identical marginals. Both evaluated on same hard costs.
    q=.5*Bgood.sum(0);ind=np.prod(np.where(f.B>0,q,1-q),axis=1);joint=np.zeros(f.S);joint[good]=.5
    np.testing.assert_allclose(ind@f.B,joint@f.B,atol=1e-14)
    save('same_marginals_example.json',dict(marginals=q.tolist(),independent_good_probability=float(ind[good].sum()),coupled_good_probability=float(joint[good].sum()),independent_costs=(ind@C).tolist(),coupled_costs=(joint@C).tolist(),interpretation='two distributions, not a proposed production sampler or evidence of full-chain impossibility'))
    for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==ph
    summary=dict(states=64,feature_limited_states=int((caps<1).sum()),automorphism_limited_states=int((auto_caps<1).sum()),extra_feature_loss_states=int((caps<auto_caps).sum()),one_target_linearly_separable_states=sum(x['one_target_linearly_separable'] for x in state_rows),cap_counts={str(c):int((caps==c).sum()) for c in np.unique(caps)},automorphism_cap_counts={str(c):int((auto_caps==c).sum()) for c in np.unique(auto_caps)},readouts=rows,all_bounds_attained_on_exact_grid=True,linear_program_count=len(lp_rows),max_terminal_reconstruction_error=max(errors),limitations=['bounds allow independent parameters per state and are relaxations of shared model','fixed archived X1 occupancy; not an upper bound over all trained reverse chains','per-step Bernoulli independence does not imply final unconditional edge independence','toy model findings do not identify production PubMed cause'])
    save('summary.json',summary);save('receipt.json',dict(status='complete',seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0,training_updates=0,new_sampled_paths=0,production_calls=0,bindings_unchanged=True));print(json.dumps({k:v for k,v in summary.items() if k!='readouts'}))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        if OUT.exists():save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
