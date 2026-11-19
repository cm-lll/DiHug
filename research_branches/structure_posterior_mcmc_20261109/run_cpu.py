"""Bounded finite posterior-inference audit. No model or production sampler imports."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv, hashlib, itertools, json, resource, signal, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/structure_posterior_mcmc_20261109'
OLD = ROOT / 'results/structure_without_edge_ce_theory_20261109'
START = time.monotonic()
def save(name, data):
    (OUT / (name + '.json')).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    if isinstance(data, list) and data and isinstance(data[0], dict):
        with (OUT / (name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, list(data[0])); w.writeheader(); w.writerows(data)
def check_budget():
    if time.monotonic()-START > 540: raise TimeoutError('Scientific computation deadline')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 2048:
        raise MemoryError('RSS budget')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(freq, target, bits, energy):
    d = (freq-target) @ bits
    return dict(TV=float(abs(freq-target).sum()/2), edge_MAE=float(abs(d).mean()),
                edge_max_error=float(abs(d).max()), energy_error=float((freq-target)@energy))
def ess(x):
    # Paired initial-positive autocorrelation sum, finite-chain diagnostic only.
    x = x.astype(float)-np.mean(x); n=len(x)
    if np.dot(x,x) < 1e-25: return None
    f=np.fft.rfft(x, n=2*n); ac=np.fft.irfft(f*np.conj(f))[:n]
    ac=ac/ac[0]; total=0.
    for j in range(1, min(n-1,1000), 2):
        pair=ac[j]+ac[j+1]
        if pair<=0: break
        total+=pair
    return float(min(n, n/max(1., 1+2*total)))

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    d=np.load(OLD/'finite_universe.npz')
    bits=d['bits']; edges=d['query'].T; families=d['families']; energy=d['energy']
    prior=d['illustrative_gibbs']; n,m=bits.shape
    assert (n,m)==(1024,10)
    lookup={tuple(e):i for i,e in enumerate(edges)}
    # Fixed templates: choosing an illegal template produces a self-loop.
    toggle=[(1<<i,0,0) for i in range(m)]
    relocate=[((1<<i)|(1<<j),1<<i,1<<j) for i,j in itertools.combinations(range(m),2) if families[i]==families[j]]
    switches=set()
    for i,j in itertools.combinations(range(m),2):
        a,b=edges[i]; c,e=edges[j]
        if families[i]!=families[j] or len({a,b,c,e})!=4: continue
        for pairs in [((a,c),(b,e)),((a,e),(b,c))]:
            g,h=[lookup[tuple(sorted(p))] for p in pairs]
            if families[g]!=families[i] or families[h]!=families[i]: continue
            left=(1<<i)|(1<<j); right=(1<<g)|(1<<h)
            lo,hi=sorted((left,right)); switches.add((left|right,lo,hi))
    templates=[toggle, relocate, sorted(switches)]
    weights=[.5,.25,.25]; names=['toggle','relocate','switch']
    retentions=[0.,.5,.95]; observations=[659,421,0,1023]
    starts=[0,1023,659]; repeats=3; steps=4000; burn=500
    checkpoints=[0,10,50,100,500,1000,4000]
    protocol=dict(start_utc=datetime.now(timezone.utc).isoformat(),
        scope='Toy training-side posterior approximation only; no models or production diffusion graphs',
        target='rho(G|x) proportional exp(-E(G))*q_t(x|G); exact previous E, tau=1, uniform base',
        asset=str(OLD/'finite_universe.npz'), asset_sha256=sha(OLD/'finite_universe.npz'),
        retentions=retentions, observations=observations, starts=starts, repeats=repeats,
        steps=steps, burn_in=burn, exact_tracking_steps=checkpoints,
        proposal_class_weights=dict(zip(names,weights)), proposal_templates=dict(zip(names,templates)),
        illegal_proposal='self-loop; never redraw until legal', MH='symmetric Q, min(1,exp(logrho_new-logrho_old))',
        seeds='chain:202611091000+condition*100+start_index*10+repeat; IID:chain+100000',
        comparator='same-count independent exact posterior draws, one for every chain',
        no_tuning=True, alpha1='Analytic delta at observation for this full-support prior; no MCMC',
        limitations='Toy energy cached; 4000 is fixed diagnostic length, not a convergence declaration. No production cost extrapolation.',
        limits=dict(threads=1,seconds=600,science_seconds=540,RSS_MiB=2048,GPU=0))
    save('protocol',protocol)  # Frozen before computing any posterior/chain results.
    save('binding',dict(source=str(Path(__file__)), source_sha256=sha(Path(__file__)),
        assets={str(OLD/f):sha(OLD/f) for f in ['finite_universe.npz','protocol.json','receipt.json']}))
    dest=[]; Q=np.zeros((n,n))
    for k,ts in enumerate(templates):
        table=np.empty((n,len(ts)),np.int32)
        for c in range(n):
            for j,(mask,p1,p2) in enumerate(ts):
                valid=k==0 or (c&mask) in (p1,p2)
                nxt=c^mask if valid else c
                table[c,j]=nxt; Q[c,nxt]+=weights[k]/len(ts)
        dest.append(table)
    assert np.max(abs(Q-Q.T))<1e-14 and np.max(abs(Q.sum(1)-1))<1e-14
    # Full enumeration of move invariants, without the previous audit's code.
    degree=np.zeros((n,5,3),int)
    for i,(u,v) in enumerate(edges):
        degree[:,u,families[i]]+=bits[:,i]; degree[:,v,families[i]]+=bits[:,i]
    for c in range(n):
        for nxt in dest[1][c]:
            assert all(bits[c,families==r].sum()==bits[nxt,families==r].sum() for r in range(3))
        for nxt in dest[2][c]: assert np.array_equal(degree[c],degree[nxt])
    save('proposal_checks',dict(symmetric_error=float(abs(Q-Q.T).max()),
        template_counts=dict(zip(names,map(len,templates))), invariants_checked_states=n,
        irreducible='all one-bit transitions positive for alpha<1; all graphs reachable',
        aperiodic='null relocation/switch proposals or MH rejection provide self-loops'))
    rows=[]; exact_rows=[]; edges_rows=[]; acceptance=[]; panel=[]; trajectories={}; posteriors={}
    priors=np.array([.25,.4,.35])[families]
    for ci,(alpha,obs) in enumerate(itertools.product(retentions,observations)):
        check_budget(); tick=time.monotonic(); x=bits[obs]
        prob=alpha*bits+(1-alpha)*priors
        logp=-energy+np.log(np.where(x,prob,1-prob)).sum(1)
        target=np.exp(logp-logp.max()); target/=target.sum(); posteriors[str(ci)]=target
        ratio=np.exp(np.minimum(0,logp[None,:]-logp[:,None]))
        P=Q*ratio; np.fill_diagonal(P,0)
        np.fill_diagonal(P,1-P.sum(1)); flow=target[:,None]*P
        db=float(abs(flow-flow.T).max()); stationary=float(abs(target@P-target).max())
        assert db<1e-13 and stationary<1e-13 and P.min()>=-1e-14
        PT=csr_matrix(P.T); dist=np.eye(n)[:,starts]; average=np.zeros_like(dist)
        for step in range(steps+1):
            if step in checkpoints:
                for si,start in enumerate(starts):
                    exact_rows.append(dict(condition=ci,retention=alpha,observation=obs,start=start,step=step,
                                           **metrics(dist[:,si],target,bits,energy)))
            if step>burn: average+=dist/(steps-burn)
            if step<steps: dist=PT@dist
            if step%500==0: check_budget()
        bias=[]
        for si,start in enumerate(starts):
            bias.append(metrics(average[:,si],target,bits,energy))
            for rep in range(repeats):
                seed=202611091000+ci*100+si*10+rep; rng=np.random.default_rng(seed)
                c=start; path=np.empty(steps,np.int16); attempts=np.zeros(3,int); null=np.zeros(3,int); accepted=np.zeros(3,int)
                for step in range(steps):
                    u=rng.random(); kind=0 if u<.5 else 1 if u<.75 else 2
                    j=int(rng.integers(len(templates[kind]))); nxt=int(dest[kind][c,j]); attempts[kind]+=1
                    draw=rng.random()  # Consume the same RNG contract even for null/rejected moves.
                    if nxt==c: null[kind]+=1
                    elif np.log(max(draw,np.finfo(float).tiny))<min(0.,logp[nxt]-logp[c]):
                        c=nxt; accepted[kind]+=1
                    path[step]=c
                kept=path[burn:]; freq=np.bincount(kept,minlength=n)/(steps-burn)
                iid=np.random.default_rng(seed+100000).choice(n,size=steps-burn,p=target)
                ifreq=np.bincount(iid,minlength=n)/(steps-burn)
                trajectories[f'c{ci}_s{si}_r{rep}']=path
                base=dict(condition=ci,retention=alpha,observation=obs,start=start,repeat=rep,seed=seed)
                rows.append(dict(**base,**metrics(freq,target,bits,energy),
                    iid_TV=metrics(ifreq,target,bits,energy)['TV'],
                    iid_edge_MAE=metrics(ifreq,target,bits,energy)['edge_MAE'],
                    energy_ESS=ess(energy[kept]), edge_count_ESS=ess(bits[kept].sum(1)),
                    distinct_states=int(len(np.unique(kept))), retained_states=len(kept)))
                for edge in range(m):
                    edges_rows.append(dict(**base,edge=edge,family=int(families[edge]),
                        exact=float(target@bits[:,edge]), estimate=float(freq@bits[:,edge]),
                        iid=float(ifreq@bits[:,edge])))
                for k,name in enumerate(names):
                    acceptance.append(dict(**base,proposal=name,attempts=int(attempts[k]),
                        null=int(null[k]),accepted_state_changes=int(accepted[k]),
                        rejected=int(attempts[k]-null[k]-accepted[k])))
        panel.append(dict(condition=ci,retention=alpha,observation=obs,detailed_balance_error=db,
            stationarity_error=stationary,exact_end_TV_max=float(max(r['TV'] for r in exact_rows if r['condition']==ci and r['step']==steps)),
            retained_expectation_TV_max=max(b['TV'] for b in bias),
            retained_expectation_edge_MAE_max=max(b['edge_MAE'] for b in bias),
            elapsed_seconds=time.monotonic()-tick))
        save('panels',panel); save('chains',rows); save('exact_tracking',exact_rows)
    save('edge_marginals',edges_rows); save('acceptance',acceptance)
    np.savez_compressed(OUT/'trajectories.npz',**trajectories)
    np.savez_compressed(OUT/'exact_posteriors.npz',**posteriors)
    save('receipt',dict(status='complete',conditions=12,chains=108,chain_transitions=432000,
        retained_MCMC_states=378000,exact_target_IID_draws=378000,model_calls=0,
        training_updates=0,production_diffusion_graphs=0,GPU_hours=0,
        seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        finished_utc=datetime.now(timezone.utc).isoformat(),formal_training_authorized=False))
if __name__=='__main__':
    try: main()
    except BaseException as e:
        if OUT.exists(): save('failure',dict(error=repr(e),seconds=time.monotonic()-START))
        raise
