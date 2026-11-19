"""Finite mathematical audit; no models, gradients through networks, or sampled graphs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time,itertools,hashlib,signal,resource,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/structure_without_edge_ce_theory_20261109'
START=time.monotonic();TYPES=np.array([0,0,1,1,1]);Q=np.array(list(itertools.combinations(range(5),2))).T
F=np.array([0 if TYPES[i]==TYPES[j]==0 else 2 if TYPES[i]==TYPES[j]==1 else 1 for i,j in Q.T]);M=10
BITS=((np.arange(1024)[:,None]>>np.arange(M))&1).astype(int)
LOOKUP={tuple(map(int,e)):i for i,e in enumerate(Q.T)}
REF_EDGES=[(0,1),(0,2),(1,2),(2,3),(3,4)]
REF=sum(1<<LOOKUP[e] for e in REF_EDGES)

def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj and isinstance(obj[0],dict):
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)

def code(b):return int(np.asarray(b,dtype=int)@(1<<np.arange(M)))
def sigmoid(s):return 1/(1+np.exp(-s))
def product(p):return np.prod(np.where(BITS,p,1-p),axis=1)
def kl(a,b):
    keep=a>0
    if np.any(b[keep]==0):return None
    return float(np.sum(a[keep]*np.log(a[keep]/b[keep])))
def entropy(p):
    x=p[p>0];return float(-np.sum(x*np.log(x)))
def rank(x):return int(np.linalg.matrix_rank(np.asarray(x,dtype=float)))

def features():
    counts=np.zeros((1024,3));deg=np.zeros((1024,5,3));tri=np.zeros((1024,4));comp=np.zeros(1024)
    for k,(u,v) in enumerate(Q.T):
        counts[:,F[k]]+=BITS[:,k];deg[:,u,F[k]]+=BITS[:,k];deg[:,v,F[k]]+=BITS[:,k]
    for u,v,w in itertools.combinations(range(5),3):
        ids=[LOOKUP[tuple(sorted(e))] for e in [(u,v),(u,w),(v,w)]]
        tri[:,int(TYPES[[u,v,w]].sum())]+=BITS[:,ids].prod(1)
    for c,b in enumerate(BITS):
        a=np.zeros((5,5),bool);a[Q[0],Q[1]]=b;a|=a.T
        unseen=set(range(5));components=0
        while unseen:
            stack=[unseen.pop()];components+=1
            while stack:
                u=stack.pop();nxt=unseen.intersection(np.flatnonzero(a[u]));unseen-=nxt;stack.extend(nxt)
        comp[c]=components
    return counts,deg,tri,comp

def energy(counts,deg,tri,comp):
    # One declared toy energy, NOT selected on results or proposed as production loss.
    number=((counts-counts[REF])/np.maximum(1.,counts[REF]))**2
    allocation=np.zeros(1024)
    for t in [0,1]:
        ids=np.flatnonzero(TYPES==t);cols=np.unique(F[np.isin(Q[0],ids)|np.isin(Q[1],ids)])
        ref=deg[REF,ids][:,cols];scale=np.maximum(1.,ref.mean(0))
        costs=[np.abs(deg[:,ids][:,:,cols]-ref[list(perm)])/scale for perm in itertools.permutations(range(len(ids)))]
        allocation+=np.min(np.array([v.mean((1,2)) for v in costs]),axis=0)/2
    closure=np.mean(((tri-tri[REF])/np.maximum(1.,tri[REF]))**2,axis=1)
    connection=(comp-comp[REF])**2
    return np.stack([number.mean(1),allocation,closure,connection],axis=1)

def switches(b):
    for e,f in itertools.combinations(np.flatnonzero(b),2):
        if F[e]!=F[f]:continue
        a,c=Q[:,e];d,h=Q[:,f]
        if len({a,c,d,h})<4:continue
        for new in [((a,d),(c,h)),((a,h),(c,d))]:
            g,j=[LOOKUP[tuple(sorted(x))] for x in new]
            if F[g]==F[j]==F[e] and not b[g] and not b[j]:yield int(e),int(f),int(g),int(j)

def union_components(edges):
    parent=list(range(1024))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for u,v in edges:
        x,y=root(u),root(v)
        if x!=y:parent[x]=y
    return len({root(i) for i in range(1024)})

def main():
    OUT.mkdir(parents=True,exist_ok=False);signal.alarm(120)
    save('protocol',dict(scope='Exact enumeration of 1024 typed five-node graphs, no graph sampling or model calls',
        node_types=TYPES.tolist(),edge_slots=Q.T.tolist(),families=F.tolist(),reference_edges=REF_EDGES,
        toy_energy='mean of relative relation-count squared error, typewise joint-degree L1 assignment, typed triangle squared error, component-count squared error; fixed unit block weights, demonstration only',
        priors=['single reference point mass','uniform type-preserving reference orbit','uniform-base Gibbs exp(-toy_energy), temperature=1 fixed illustration'],
        corruption_retention=[0.,.5,.95,1.],corruption_family_priors=[.25,.4,.35],
        observations='reference, first distinct type-permuted reference, empty, full; predeclared, no effect selection',
        tests='operation spans/components, fixed-background zero-sum, all-toggle exact gradient, two-slot collapse, mixed-objective loop integrability, posterior normalization/marginals/factorization/endpoints',
        limits=dict(threads=1,seconds=120,RSS_MiB=2048,GPU=0)))
    counts,deg,tri,comp=features();parts=energy(counts,deg,tri,comp);E=parts.mean(1)
    assert E[REF]==0
    orbit=[]
    for aa in itertools.permutations([0,1]):
        for bb in itertools.permutations([2,3,4]):
            perm=np.array(aa+bb);dest=np.zeros(M,int)
            for k,(u,v) in enumerate(Q.T):dest[LOOKUP[tuple(sorted((perm[u],perm[v])))]]=BITS[REF,k]
            orbit.append(code(dest))
    orbit=sorted(set(orbit));assert np.max(E[orbit])==0
    # Exhaustive invariance under all 12 type-preserving permutations.
    permutation_checks=0
    for aa in itertools.permutations([0,1]):
        for bb in itertools.permutations([2,3,4]):
            perm=np.array(aa+bb);dest=np.zeros_like(BITS)
            for k,(u,v) in enumerate(Q.T):dest[:,LOOKUP[tuple(sorted((perm[u],perm[v])))]]=BITS[:,k]
            mapped=dest@(1<<np.arange(M));assert np.max(abs(E-E[mapped]))<1e-12;permutation_checks+=1024
    vectors={'birth_death':list(np.eye(M)), 'relocation':[], 'switch':[]};edges={k:[] for k in vectors}
    for e,f in itertools.combinations(range(M),2):
        if F[e]==F[f]:v=np.zeros(M);v[e]=-1;v[f]=1;vectors['relocation'].append(v)
    cycle_max=0.;witness=None;cycle_count=0;switch_occurrences=0
    for c,b in enumerate(BITS):
        for e in range(M):edges['birth_death'].append((c,c^(1<<e)))
        for e in np.flatnonzero(b):
            for f in np.flatnonzero((b==0)&(F==F[e])):edges['relocation'].append((c,c^(1<<int(e))^(1<<int(f))))
        for e,f,g,h in switches(b):
            d=c^(1<<e)^(1<<f)^(1<<g)^(1<<h);v=np.zeros(M);v[[e,f]]=-1;v[[g,h]]=1
            vectors['switch'].append(v);edges['switch'].append((c,d));switch_occurrences+=1
            assert np.array_equal(deg[c],deg[d])
            for j in set(range(M))-{e,f,g,h}:
                c1=c^(1<<j);d1=d^(1<<j)
                # If switches use closure, flips only number/allocation, these preferences need not integrate.
                alloc=parts[:,0]+parts[:,1];closure=parts[:,2]+parts[:,3]
                circ=(closure[d]-closure[c])+(alloc[d1]-alloc[d])+(closure[c1]-closure[d1])+(alloc[c]-alloc[c1])
                unified=(E[d]-E[c])+(E[d1]-E[d])+(E[c1]-E[d1])+(E[c]-E[c1]);assert abs(unified)<1e-12
                cycle_count+=1
                if abs(circ)>cycle_max:
                    cycle_max=abs(float(circ));witness=dict(graph_codes=[c,d,d1,c1,c],operation_types=['switch','toggle','switch','toggle'],split_energy_circulation=float(circ),unified_energy_circulation=float(unified))
    summary=[]
    incidence=np.zeros((15,M))
    for k,(u,v) in enumerate(Q.T):incidence[3*u+F[k],k]=incidence[3*v+F[k],k]=1
    for name in vectors:
        V=np.array(vectors[name]);summary.append(dict(operations=name,displacement_rank=rank(V),graph_space_components=union_components(edges[name]),edges_enumerated=len(edges[name]),
            max_node_relation_degree_change=float(abs(V@incidence.T).max())))
    assert rank(vectors['birth_death'])==10 and rank(vectors['relocation'])==7 and rank(vectors['switch'])==2
    save('operation_coverage',summary);save('integrability',dict(cycles=cycle_count,max_split_energy_circulation=cycle_max,witness=witness,
        meaning='Different per-action energies need not define one target graph density. This does not say a sum of parameter losses is mathematically undefined.'))
    # Exact expected-hard-energy gradient under product Bernoulli; finite differences and conditional increments.
    rng=np.random.default_rng(2026110902);checks=[]
    for trial in range(12):
        z=rng.uniform(-2,2,M);p=sigmoid(z);weights=product(p);grad=weights@(E[:,None]*(BITS-p))
        conditional=np.zeros(M)
        for e in range(M):
            mask=BITS[:,e]==0;base_prob=weights[mask]/(1-p[e]);codes=np.flatnonzero(mask)
            conditional[e]=p[e]*(1-p[e])*np.sum(base_prob*(E[codes^(1<<e)]-E[codes]))
        assert np.max(abs(grad-conditional))<1e-12
        fd=[]
        for e in range(M):
            zz=z.copy();zz[e]+=1e-5;za=z.copy();za[e]-=1e-5
            fd.append((product(sigmoid(zz))@E-product(sigmoid(za))@E)/2e-5)
        error=float(np.max(abs(grad-fd)));assert error<1e-9
        checks.append(dict(trial=trial,exact_vs_increment_error=float(abs(grad-conditional).max()),fd_error=error))
    save('exact_risk_gradient',checks)
    save('two_slot_example',dict(target='one edge between a center and two exchangeable leaves',energy='(edge_count-1)^2',
        independent_half_expected_energy=.5,deterministic_one_edge_energy=0.,correlated_uniform_exact_one_energy=0.,
        independent_half_energy_gradient=[0.,0.],meaning='expected energy alone neither identifies desired uncertainty nor excludes deterministic minimizers'))
    # Three explicitly defined prior distributions, no trained models.
    prior_ref=np.zeros(1024);prior_ref[REF]=1
    prior_orbit=np.zeros(1024);prior_orbit[orbit]=1/len(orbit)
    prior_gibbs=np.exp(-E);prior_gibbs/=prior_gibbs.sum()
    priors={'reference_point':prior_ref,'type_orbit':prior_orbit,'illustrative_gibbs':prior_gibbs}
    obs=[REF,next(c for c in orbit if c!=REF),0,1023];pi=np.array([.25,.4,.35])[F]
    panels=[];per_edge=[];all_post=[];marginal_pairs=[]
    for name,prior in priors.items():
        unconditional=prior@BITS
        for a in [0.,.5,.95,1.]:
            ps=[]
            for x in obs:
                prob1=a*BITS+(1-a)*pi;like=np.prod(np.where(BITS[x],prob1,1-prob1),axis=1)
                unnorm=prior*like;mass=float(unnorm.sum())
                row=dict(prior=name,retention=a,observation_code=x,evidence=mass,defined=mass>0)
                if not mass:
                    panels.append(dict(**row,posterior_entropy=None,posterior_mean_energy=None,product_marginal_mean_energy=None,posterior_vs_product_TV=None,posterior_vs_product_KL=None,product_sampling_exact=False,soft_label_CE_optimum=None,unconditional_prediction_CE=None,conditional_odds_logit_range_max=None));continue
                posterior=unnorm/mass;marg=posterior@BITS;pr=product(marg)
                assert abs(posterior.sum()-1)<1e-12 and abs(pr.sum()-1)<1e-12 and np.max(abs(pr@BITS-marg))<1e-12
                if a==0:assert np.max(abs(posterior-prior))<1e-12
                if a==1:assert posterior[x]==1.
                ce_opt=float(np.mean([-m*np.log(m)-(1-m)*np.log1p(-m) if 0<m<1 else 0. for m in marg]))
                baseline=[]
                for m,v in zip(marg,unconditional):
                    if (v==0 and m>0) or (v==1 and m<1):baseline=None;break
                    baseline.append(-(m*np.log(v) if m>0 else 0.)-((1-m)*np.log1p(-v) if m<1 else 0.))
                ce_base=float(np.mean(baseline)) if baseline is not None else None
                if ce_base is not None:assert ce_base>=ce_opt-1e-12
                ranges=[]
                for e in range(M):
                    off=np.flatnonzero(BITS[:,e]==0);on=off^(1<<e);valid=(posterior[off]>0)&(posterior[on]>0)
                    if valid.any():
                        odds=np.log(posterior[on[valid]])-np.log(posterior[off[valid]]);ranges.append(float(np.ptp(odds)))
                    per_edge.append(dict(prior=name,retention=a,observation_code=x,slot=e,relation=int(F[e]),reference_label=int(BITS[REF,e]),
                        observation_label=int(BITS[x,e]),posterior_clean_probability=float(marg[e]),prior_probability=float(unconditional[e])))
                panels.append(dict(**row,posterior_entropy=entropy(posterior),posterior_mean_energy=float(posterior@E),product_marginal_mean_energy=float(pr@E),
                    posterior_vs_product_TV=float(abs(posterior-pr).sum()/2),posterior_vs_product_KL=kl(posterior,pr),product_sampling_exact=bool(np.max(abs(pr-posterior))<1e-12),
                    soft_label_CE_optimum=ce_opt,unconditional_prediction_CE=ce_base,conditional_odds_logit_range_max=max(ranges) if ranges else None))
                ps.append((x,marg));all_post.append(dict(prior=name,retention=a,observation_code=x,posterior=posterior.tolist()))
            for (x,p),(xx,pp) in itertools.combinations(ps,2):marginal_pairs.append(dict(prior=name,retention=a,observation0=x,observation1=xx,mean_abs_marginal_difference=float(abs(p-pp).mean())))
    save('posterior_panels',panels);save('posterior_edges',per_edge);save('observation_dependence',marginal_pairs)
    np.savez_compressed(OUT/'finite_universe.npz',bits=BITS,types=TYPES,query=Q,families=F,reference=BITS[REF],energy_parts=parts,energy=E,orbit=np.array(orbit),**priors)
    # Toy exact arrays are finite sums, not sampled graphs or generated scientific assets.
    save('exact_posterior_distributions',all_post)
    sources=[Path(__file__),ROOT/'experiments/seed_coverage_repair_v1/package/hsgd/kernel.py',ROOT/'research_branches/degree_preserving_switch_20261109/run_cpu.py',
        ROOT/'sparse_diffusion/structure_energy/mixed_edit_diffusion.py',ROOT/'sparse_diffusion/structure_energy/role_guided_edit_diffusion.py']
    save('binding',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    receipt=dict(status='complete',graphs_enumerated=1024,reference_code=REF,orbit_size=len(orbit),zero_energy_graphs=int((E==0).sum()),
        permutation_checks=permutation_checks,switch_occurrences=switch_occurrences,posterior_requested=len(panels),posterior_defined=sum(r['defined'] for r in panels),
        posterior_undefined=sum(not r['defined'] for r in panels),gradient_coordinates=12*M,
        max_exact_risk_fd_error=max(r['fd_error'] for r in checks),seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_calls=0,GPU_hours=0,training_updates=0,new_sampled_graphs=0,scientific_config_changed=False)
    assert receipt['RSS_MiB']<2048;save('receipt',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        if OUT.exists():save('failure',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
